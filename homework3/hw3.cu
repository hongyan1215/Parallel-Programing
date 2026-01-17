#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>

#define pi 3.1415926535897932384626433832795f

// ===== CUDA-compatible vector structures and math functions =====

__device__ __host__ inline float3 make_float3(float v) {
    return make_float3(v, v, v);
}

// Vector operators
__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__device__ __host__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __host__ inline float2 operator*(const float2& a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__device__ __host__ inline float2 operator/(const float2& a, float s) {
    return make_float2(a.x / s, a.y / s);
}

// Math functions - use fast intrinsics where possible
__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float len = length(v);
    float inv_len = 1.0f / len;  // Use reciprocal instead of division
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

__device__ __forceinline__ float clamp(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__device__ __forceinline__ float3 clamp(const float3& v, float minVal, float maxVal) {
    return make_float3(clamp(v.x, minVal, maxVal),
                       clamp(v.y, minVal, maxVal),
                       clamp(v.z, minVal, maxVal));
}

__device__ __forceinline__ float custom_min(float a, float b) {
    return a < b ? a : b;
}

__device__ __forceinline__ float custom_pow(float base, float exp) {
    return __powf(base, exp);  // Use fast intrinsic
}

__device__ __forceinline__ float3 custom_pow(const float3& v, float exp) {
    return make_float3(__powf(v.x, exp), __powf(v.y, exp), __powf(v.z, exp));
}

// Matrix-vector multiplication for rotation
__device__ __forceinline__ float3 mat_mul_vec(float m[3][3], const float3& v) {
    return make_float3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
    );
}

// ===== Constants =====
__constant__ float c_power = 8.0f;
__constant__ int c_md_iter = 24;
__constant__ int c_ray_step = 10000;
__constant__ int c_shadow_step = 1500;
__constant__ float c_step_limiter = 0.2f;
__constant__ float c_ray_multiplier = 0.1f;
__constant__ float c_bailout = 2.0f;
__constant__ float c_eps = 0.0005f;
__constant__ float c_FOV = 1.5f;
__constant__ float c_far_plane = 100.0f;
__constant__ int c_AA = 3;

// Precomputed rotation matrix (90 degrees around X-axis)
__constant__ float c_rotation_matrix[3][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 0.0f}
};

// ===== Device functions =====

// Palette function
__device__ __forceinline__ float3 pal(float t, const float3& a, const float3& b,
                                      const float3& c, const float3& d) {
    return make_float3(
        a.x + b.x * __cosf(2.0f * pi * (c.x * t + d.x)),
        a.y + b.y * __cosf(2.0f * pi * (c.y * t + d.y)),
        a.z + b.z * __cosf(2.0f * pi * (c.z * t + d.z))
    );
}

// Mandelbulb distance function
__device__ float md(const float3& p, float& trap) {
    float3 v = p;
    float dr = 1.0f;
    float r = length(v);
    trap = r;

    #pragma unroll 4  // Partial unroll for better performance
    for (int i = 0; i < c_md_iter; ++i) {
        float theta = atan2f(v.y, v.x) * c_power;
        float phi = asinf(v.z / r) * c_power;
        dr = c_power * __powf(r, c_power - 1.0f) * dr + 1.0f;

        float r_pow = __powf(r, c_power);
        float cos_theta = __cosf(theta);
        float sin_theta = __sinf(theta);
        float cos_phi = __cosf(phi);
        float sin_phi = __sinf(phi);

        v = p + r_pow * make_float3(cos_theta * cos_phi, cos_phi * sin_theta, -sin_phi);

        trap = custom_min(trap, r);

        r = length(v);
        if (r > c_bailout) break;
    }
    return 0.5f * __logf(r) * r / dr;
}

// Scene mapping
__device__ __forceinline__ float map(const float3& p, float& trap, int& ID) {
    float3 rp = mat_mul_vec(c_rotation_matrix, p);
    ID = 1;
    return md(rp, trap);
}

// Overload without trap and ID
__device__ __forceinline__ float map(const float3& p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// Surface normal calculation
__device__ float3 calcNor(const float3& p) {
    float2 e = make_float2(c_eps, 0.0f);
    float3 ex = make_float3(e.x, e.y, e.y);
    float3 ey = make_float3(e.y, e.x, e.y);
    float3 ez = make_float3(e.y, e.y, e.x);

    return normalize(make_float3(
        map(p + ex) - map(p - ex),
        map(p + ey) - map(p - ey),
        map(p + ez) - map(p - ez)
    ));
}

// Soft shadow with adaptive step (more conservative than original clamp)
__device__ float softshadow(const float3& ro, const float3& rd, float k) {
    float res = 1.0f;
    float t = 0.001f;

    for (int i = 0; i < c_shadow_step; ++i) {
        float h = map(make_float3(ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t));
        res = fminf(res, k * h / t);

        if (res < 0.02f || t > 20.0f) break;

        // Adaptive step with bounds similar to original
        t += clamp(h, 0.001f, c_step_limiter * 2.5f);  // Allow up to 0.5 instead of 0.2
    }
    return clamp(res, 0.02f, 1.0f);
}

// Ray tracing
__device__ float trace(const float3& ro, const float3& rd, float& trap, int& ID) {
    float t = 0.0f;
    float len = 0.0f;

    #pragma unroll 2
    for (int i = 0; i < c_ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (fabsf(len) < c_eps || t > c_far_plane) break;
        t += len * c_ray_multiplier;
    }
    return t < c_far_plane ? t : -1.0f;
}

// ===== Main rendering kernel with optimizations =====
__global__ void __launch_bounds__(256, 2)
render_kernel(uchar4* image,
              unsigned int width, unsigned int height,
              float3 camera_pos, float3 target_pos) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= width || i >= height) return;

    float2 iResolution = make_float2((float)width, (float)height);

    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;

    // Precompute camera basis vectors (shared across AA samples)
    float3 cf = normalize(target_pos - camera_pos);
    float3 cs = normalize(cross(cf, make_float3(0.0f, 1.0f, 0.0f)));
    float3 cu = normalize(cross(cs, cf));
    float3 sd = normalize(camera_pos);  // Sun direction

    // Anti-aliasing
    #pragma unroll
    for (int m = 0; m < c_AA; ++m) {
        #pragma unroll
        for (int n = 0; n < c_AA; ++n) {
            float2 p = make_float2((float)j, (float)i) + make_float2((float)m, (float)n) / (float)c_AA;

            // Convert to normalized device coordinates
            float2 uv = (make_float2(-iResolution.x, -iResolution.y) + p * 2.0f) / iResolution.y;
            uv.y *= -1.0f;

            // Camera setup
            float3 ro = camera_pos;
            float3 rd = normalize(cs * uv.x + cu * uv.y + cf * c_FOV);

            // Ray marching
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);

            // Lighting
            float3 col = make_float3(0.0f);
            float3 sc = make_float3(1.0f, 0.9f, 0.717f);

            // Coloring
            if (d < 0.0f) {
                col = make_float3(0.0f);
            } else {
                float3 pos = ro + rd * d;
                float3 nr = calcNor(pos);
                float3 hal = normalize(sd - rd);

                col = pal(trap - 0.4f, make_float3(0.5f), make_float3(0.5f),
                         make_float3(1.0f), make_float3(0.0f, 0.1f, 0.2f));
                float3 ambc = make_float3(0.3f);
                float gloss = 32.0f;

                float amb = (0.7f + 0.3f * nr.y) *
                            (0.2f + 0.8f * clamp(0.05f * __logf(trap), 0.0f, 1.0f));
                float sdw = softshadow(pos + nr * 0.001f, sd, 16.0f);
                float dif = clamp(dot(sd, nr), 0.0f, 1.0f) * sdw;
                float spe = __powf(clamp(dot(nr, hal), 0.0f, 1.0f), gloss) * dif;

                float3 lin = make_float3(0.0f);
                lin = lin + ambc * (0.05f + 0.95f * amb);
                lin = lin + sc * dif * 0.8f;
                col = col * lin;

                col = make_float3(
                    __powf(col.x, 0.7f),
                    __powf(col.y, 0.9f),
                    __powf(col.z, 1.0f)
                );
                col = col + make_float3(spe * 0.8f);
            }

            col = clamp(custom_pow(col, 0.4545f), 0.0f, 1.0f);
            fcol_r += col.x;
            fcol_g += col.y;
            fcol_b += col.z;
        }
    }

    // Average and convert to unsigned char
    float inv_aa = 1.0f / (float)(c_AA * c_AA);
    fcol_r *= inv_aa * 255.0f;
    fcol_g *= inv_aa * 255.0f;
    fcol_b *= inv_aa * 255.0f;

    // Use vectorized uchar4 store for 16-byte aligned write
    int idx = i * width + j;
    image[idx] = make_uchar4((unsigned char)fcol_r,
                             (unsigned char)fcol_g,
                             (unsigned char)fcol_b,
                             255);
}

// ===== Host code =====

void write_png(const char* filename, unsigned char* image,
               unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    assert(argc == 10);

    // Parse arguments
    float3 camera_pos = make_float3((float)atof(argv[1]), (float)atof(argv[2]), (float)atof(argv[3]));
    float3 target_pos = make_float3((float)atof(argv[4]), (float)atof(argv[5]), (float)atof(argv[6]));
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);

    // Allocate host memory
    unsigned char* h_image = new unsigned char[width * height * 4];

    // Allocate device memory as uchar4 for aligned access
    uchar4* d_image;
    cudaMalloc(&d_image, width * height * sizeof(uchar4));

    // Configure L1 cache preference for high-register kernels
    cudaFuncSetCacheConfig(render_kernel, cudaFuncCachePreferL1);

    // Setup kernel configuration with larger blocks for better occupancy
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    render_kernel<<<gridDim, blockDim>>>(d_image, width, height, camera_pos, target_pos);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel Time: %.3f ms (%.3f seconds)\n", milliseconds, milliseconds / 1000.0f);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(h_image, d_image, width * height * sizeof(uchar4),
               cudaMemcpyDeviceToHost);

    // Save image
    write_png(argv[9], h_image, width, height);

    // Cleanup
    delete[] h_image;
    cudaFree(d_image);

    return 0;
}
