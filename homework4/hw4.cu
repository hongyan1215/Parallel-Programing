//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <vector>

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <cassert>
#include <chrono>

#include <cuda_runtime.h>

#include "sha256.h"

#ifndef GPU_CONSTANT_HEADER_SIZE
#define GPU_CONSTANT_HEADER_SIZE 76
#endif

#ifndef GPU_CONSTANT_HEADER_WORDS
#define GPU_CONSTANT_HEADER_WORDS (GPU_CONSTANT_HEADER_SIZE / sizeof(unsigned int))
#endif

__constant__ unsigned int c_block_header_words[GPU_CONSTANT_HEADER_WORDS];
__constant__ unsigned char c_target[32];

extern __constant__ unsigned int k_device[64];

static unsigned char HEX_DECODE_TABLE[256];

struct HexDecodeTableInit
{
    HexDecodeTableInit()
    {
        for (int i = 0; i < 256; ++i)
        {
            HEX_DECODE_TABLE[i] = 0;
        }

        for (int value = 0; value <= 9; ++value)
        {
            HEX_DECODE_TABLE['0' + value] = static_cast<unsigned char>(value);
        }

        for (int value = 0; value < 6; ++value)
        {
            HEX_DECODE_TABLE['a' + value] = static_cast<unsigned char>(10 + value);
            HEX_DECODE_TABLE['A' + value] = static_cast<unsigned char>(10 + value);
        }
    }
};

static const HexDecodeTableInit HEX_DECODE_TABLE_INIT{};

#ifdef __CUDACC__
#define HW4_HD __host__ __device__
#else
#define HW4_HD
#endif

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                        \
        cudaError_t cuda_status = (call);                                                       \
        if (cuda_status != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_status)); \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;

HW4_HD void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len);

////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
inline unsigned char decode(unsigned char c)
{
    return HEX_DECODE_TABLE[c];
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    for(size_t s = 0, b = string_len/2 - 1; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

HW4_HD int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

__device__ __forceinline__ unsigned int rotr32(const unsigned int x, const unsigned int n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ unsigned int bswap32(const unsigned int x)
{
    return (x >> 24) |
           ((x & 0x00FF0000u) >> 8) |
           ((x & 0x0000FF00u) << 8) |
           (x << 24);
}

__device__ __forceinline__ void sha256_process_block(const unsigned int input[16], unsigned int (&state)[8])
{
    unsigned int w[64];

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        w[i] = input[i];
    }

#pragma unroll
    for (int i = 16; i < 64; ++i)
    {
        unsigned int s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        unsigned int s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    unsigned int a = state[0];
    unsigned int b = state[1];
    unsigned int c = state[2];
    unsigned int d = state[3];
    unsigned int e = state[4];
    unsigned int f = state[5];
    unsigned int g = state[6];
    unsigned int h = state[7];

#pragma unroll 64
    for (int i = 0; i < 64; ++i)
    {
        unsigned int S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int temp1 = h + S1 + ch + k_device[i] + w[i];
        unsigned int S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        unsigned int maj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ __forceinline__ void double_sha256_inline(const unsigned int header_words[GPU_CONSTANT_HEADER_WORDS + 1],
                                                     unsigned int (&hash_le)[8])
{
    unsigned int block[16];
    unsigned int state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        block[i] = bswap32(header_words[i]);
    }
    sha256_process_block(block, state);

    block[0] = bswap32(header_words[16]);
    block[1] = bswap32(header_words[17]);
    block[2] = bswap32(header_words[18]);
    block[3] = bswap32(header_words[GPU_CONSTANT_HEADER_WORDS]);
    block[4] = 0x80000000u;
#pragma unroll
    for (int i = 5; i < 15; ++i)
    {
        block[i] = 0u;
    }
    block[15] = 80u * 8u;
    sha256_process_block(block, state);

    unsigned int state2[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        block[i] = state[i];
    }
    block[8] = 0x80000000u;
#pragma unroll
    for (int i = 9; i < 15; ++i)
    {
        block[i] = 0u;
    }
    block[15] = 32u * 8u;

    sha256_process_block(block, state2);

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        hash_le[i] = bswap32(state2[i]);
    }
}

struct GpuSearchStats {
    unsigned long long hashes_checked;
    float gpu_ms;
    double wall_ms;

    GpuSearchStats() : hashes_checked(0ULL), gpu_ms(0.0f), wall_ms(0.0) {}
};

__global__ void gpu_mine_kernel(unsigned int start_nonce,
                                unsigned int end_nonce,
                                unsigned int batch_stride,
                                unsigned int *result_nonce,
                                int *found_flag,
                                unsigned char *result_hash)
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned long long nonce = static_cast<unsigned long long>(start_nonce) + thread_id;
    const unsigned long long limit = static_cast<unsigned long long>(end_nonce);

    if (nonce > limit)
    {
        return;
    }

    unsigned int header_words[GPU_CONSTANT_HEADER_WORDS + 1];
#pragma unroll
    for (int i = 0; i < GPU_CONSTANT_HEADER_WORDS; ++i)
    {
        header_words[i] = c_block_header_words[i];
    }

    const unsigned int thread_batch = batch_stride;

    while (nonce <= limit)
    {
        if (atomicAdd(found_flag, 0) != 0)
        {
            break;
        }

        unsigned long long current_nonce = nonce;

#pragma unroll
        for (unsigned int step = 0; step < thread_batch && current_nonce <= limit; ++step)
        {
            if (atomicAdd(found_flag, 0) != 0)
            {
                break;
            }

            header_words[GPU_CONSTANT_HEADER_WORDS] = static_cast<unsigned int>(current_nonce);

            unsigned int hash_le[8];
            double_sha256_inline(header_words, hash_le);

            if (little_endian_bit_comparison(reinterpret_cast<unsigned char*>(hash_le), c_target, 32) < 0)
            {
                if (atomicCAS(found_flag, 0, 1) == 0)
                {
                    *result_nonce = static_cast<unsigned int>(current_nonce);
                    unsigned char *hash_bytes = reinterpret_cast<unsigned char*>(hash_le);
#pragma unroll
                    for (int i = 0; i < 32; ++i)
                    {
                        result_hash[i] = hash_bytes[i];
                    }
                }
                return;
            }

            current_nonce += stride;
        }

        nonce = current_nonce;
    }
}

bool find_nonce_gpu(const HashBlock &block_template,
                    const unsigned char *target,
                    unsigned int &found_nonce,
                    SHA256 &found_hash,
                    GpuSearchStats &stats)
{
    static unsigned int cached_header_words[GPU_CONSTANT_HEADER_WORDS];
    static unsigned char cached_target[32];
    static bool cache_valid = false;
    constexpr int PIPELINE_DEPTH = 2;
    struct PipelineSlot {
        cudaStream_t stream;
        unsigned int *d_result_nonce;
        unsigned char *d_result_hash;
        int *d_found_flag;
        cudaEvent_t completion_event;
        bool pending;
    };
    static PipelineSlot slots[PIPELINE_DEPTH];
    static bool slots_initialized = false;

    unsigned int header_words_host[GPU_CONSTANT_HEADER_WORDS];
    memcpy(header_words_host, &block_template, GPU_CONSTANT_HEADER_SIZE);

    bool header_changed = !cache_valid || memcmp(cached_header_words, header_words_host, sizeof(header_words_host)) != 0;
    bool target_changed = !cache_valid || memcmp(cached_target, target, 32) != 0;

    if (!slots_initialized)
    {
        for (int i = 0; i < PIPELINE_DEPTH; ++i)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&slots[i].d_result_nonce, sizeof(unsigned int)));
            CUDA_CHECK(cudaMalloc(&slots[i].d_result_hash, 32));
            CUDA_CHECK(cudaMalloc(&slots[i].d_found_flag, sizeof(int)));
            CUDA_CHECK(cudaEventCreateWithFlags(&slots[i].completion_event, cudaEventDisableTiming));
            slots[i].pending = false;
        }
        slots_initialized = true;
    }

    // use stream 0 for constant uploads
    cudaStream_t upload_stream = slots[0].stream;

    if (header_changed)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_block_header_words,
                                           header_words_host,
                                           GPU_CONSTANT_HEADER_SIZE,
                                           0,
                                           cudaMemcpyHostToDevice,
                                           upload_stream));
        memcpy(cached_header_words, header_words_host, sizeof(header_words_host));
    }

    if (target_changed)
    {
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_target,
                                           target,
                                           32,
                                           0,
                                           cudaMemcpyHostToDevice,
                                           upload_stream));
        memcpy(cached_target, target, 32);
    }

    cache_valid = true;

    if (header_changed || target_changed)
    {
        CUDA_CHECK(cudaStreamSynchronize(upload_stream));
    }

    const unsigned int threads_per_block = 512;
    const unsigned int num_blocks = 192;

    auto wall_start = std::chrono::steady_clock::now();

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event, 0));

    bool found = false;
    stats.hashes_checked = 0ULL;
    stats.gpu_ms = 0.0f;
    stats.wall_ms = 0.0;
    found_nonce = 0;

    const unsigned long long total_threads = static_cast<unsigned long long>(threads_per_block) * num_blocks;
    const unsigned int iterations_per_thread = 512;
    const unsigned long long batch_size = total_threads * static_cast<unsigned long long>(iterations_per_thread);

    auto try_finalize_slot = [&](int idx, bool wait) {
        PipelineSlot &slot = slots[idx];
        if (!slot.pending)
        {
            return false;
        }
        cudaError_t status = wait ? cudaEventSynchronize(slot.completion_event) : cudaEventQuery(slot.completion_event);
        if (status == cudaErrorNotReady)
        {
            return false;
        }
        CUDA_CHECK(status);

        int h_found_flag = 0;
        CUDA_CHECK(cudaMemcpy(&h_found_flag, slot.d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_found_flag != 0 && !found)
        {
            CUDA_CHECK(cudaMemcpy(&found_nonce, slot.d_result_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(found_hash.b, slot.d_result_hash, 32, cudaMemcpyDeviceToHost));
            found = true;
        }
        slot.pending = false;
        return true;
    };

    auto pending_exists = [&]() {
        for (int i = 0; i < PIPELINE_DEPTH; ++i)
        {
            if (slots[i].pending)
            {
                return true;
            }
        }
        return false;
    };

    unsigned long long next_start = 0ULL;
    while ((next_start <= 0xffffffffULL && !found) || pending_exists())
    {
        // poll pending slots
        for (int i = 0; i < PIPELINE_DEPTH && !found; ++i)
        {
            try_finalize_slot(i, false);
        }

        if (found)
        {
            // drain remaining slots to keep buffers consistent
            for (int i = 0; i < PIPELINE_DEPTH; ++i)
            {
                try_finalize_slot(i, true);
            }
            break;
        }

        if (next_start > 0xffffffffULL)
        {
            bool waited = false;
            for (int i = 0; i < PIPELINE_DEPTH; ++i)
            {
                if (try_finalize_slot(i, true))
                {
                    waited = true;
                    break;
                }
            }
            if (!waited)
            {
                break;
            }
            continue;
        }

        int free_slot = -1;
        for (int i = 0; i < PIPELINE_DEPTH; ++i)
        {
            if (!slots[i].pending)
            {
                free_slot = i;
                break;
            }
        }

        if (free_slot == -1)
        {
            // wait for one slot to free up
            for (int i = 0; i < PIPELINE_DEPTH; ++i)
            {
                if (slots[i].pending)
                {
                    try_finalize_slot(i, true);
                    break;
                }
            }
            continue;
        }

        unsigned long long batch_end = next_start + batch_size - 1ULL;
        if (batch_end > 0xffffffffULL)
        {
            batch_end = 0xffffffffULL;
        }

        unsigned int start_nonce = static_cast<unsigned int>(next_start);
        unsigned int end_nonce = static_cast<unsigned int>(batch_end);

        PipelineSlot &slot = slots[free_slot];
        CUDA_CHECK(cudaMemsetAsync(slot.d_found_flag, 0, sizeof(int), slot.stream));
        unsigned int zero_nonce = 0;
        CUDA_CHECK(cudaMemcpyAsync(slot.d_result_nonce, &zero_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice, slot.stream));
        CUDA_CHECK(cudaMemsetAsync(slot.d_result_hash, 0, 32, slot.stream));

        gpu_mine_kernel<<<num_blocks, threads_per_block, 0, slot.stream>>>(start_nonce,
                                                                           end_nonce,
                                                                           iterations_per_thread,
                                                                           slot.d_result_nonce,
                                                                           slot.d_found_flag,
                                                                           slot.d_result_hash);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(slot.completion_event, slot.stream));
        slot.pending = true;

        stats.hashes_checked += (batch_end - next_start + 1ULL);
        next_start = batch_end + 1ULL;
    }

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&stats.gpu_ms, start_event, stop_event));

    auto wall_end = std::chrono::steady_clock::now();
    stats.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    return found;
}

bool find_nonce_cpu(HashBlock &block,
                    const unsigned char *target,
                    SHA256 &hash_out,
                    unsigned long long &checked)
{
    checked = 0ULL;
    SHA256 ctx;

    for (block.nonce = 0x00000000; ; ++block.nonce)
    {
        double_sha256(&ctx, reinterpret_cast<unsigned char*>(&block), sizeof(HashBlock));
        ++checked;

        if (little_endian_bit_comparison(ctx.b, target, 32) < 0)
        {
            hash_out = ctx;
            return true;
        }

        if (block.nonce == 0xffffffff)
        {
            break;
        }
    }

    return false;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

HW4_HD void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    static std::vector<unsigned char> raw_storage;
    static std::vector<unsigned char*> pointer_storage;

    raw_storage.resize((total_count + 1) * 32);
    pointer_storage.resize(total_count + 1);

    unsigned char *raw_list = raw_storage.data();
    unsigned char **list = pointer_storage.data();

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);
}


void solve(FILE *fin, FILE *fout)
{

    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    static std::vector<char> merkle_storage;
    static std::vector<char*> merkle_ptrs;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    printf("start hashing");

    merkle_storage.resize(tx * 65);
    merkle_ptrs.resize(tx);
    for(int i=0;i<tx;++i)
    {
        merkle_ptrs[i] = merkle_storage.data() + i * 65;
        getline(merkle_ptrs[i], 65, fin);
        merkle_ptrs[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_ptrs.data());

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");


    // **** solve block ****
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: "); print_hex_inverse(merkle_root, 32); printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash,                  prevhash,    64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime,   ntime,     8);
    block.nonce = 0;
    
    
    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    
    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");


    // ********** find nonce **************
    printf("Launching GPU nonce search...\n");

    GpuSearchStats gpu_stats;
    unsigned int found_nonce = 0;
    SHA256 gpu_hash = {};
    bool gpu_found = find_nonce_gpu(block, target_hex, found_nonce, gpu_hash, gpu_stats);

    SHA256 final_hash = {};
    bool solution_found = gpu_found;

    if (gpu_found)
    {
        block.nonce = found_nonce;
        double_sha256(&final_hash, reinterpret_cast<unsigned char*>(&block), sizeof(block));

        if (memcmp(final_hash.b, gpu_hash.b, 32) != 0)
        {
            printf("Warning: host verification hash differs from GPU result.\n");
        }

        printf("GPU search found nonce %u\n", block.nonce);
    }
    else
    {
        printf("GPU search did not find a nonce, falling back to CPU search...\n");
        unsigned long long cpu_checked = 0ULL;
        auto cpu_start = std::chrono::steady_clock::now();
        bool cpu_found = find_nonce_cpu(block, target_hex, final_hash, cpu_checked);
        auto cpu_end = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        if (!cpu_found)
        {
            fprintf(stderr, "Failed to find a valid nonce using GPU or CPU.\n");
            return;
        }

        printf("CPU search checked %llu hashes in %.3f ms\n", cpu_checked, cpu_ms);
        solution_found = true;
    }

    printf("GPU hashes scheduled: %llu\n", gpu_stats.hashes_checked);
    printf("GPU kernel time: %.3f ms\n", gpu_stats.gpu_ms);
    printf("GPU total wall time: %.3f ms\n\n", gpu_stats.wall_ms);

    if (!solution_found)
    {
        fprintf(stderr, "No valid nonce found after GPU/CPU search.\n");
        return;
    }

    // print result
    printf("Found Solution!!\n");
    printf("Nonce: %u\n", block.nonce);

    printf("hash(little): ");
    print_hex(final_hash.b, 32);
    printf("\n");

    printf("hash(big):    ");
    print_hex_inverse(final_hash.b, 32);
    printf("\n\n");

    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i)
    {
        solve(fin, fout);
    }

    return 0;
}
