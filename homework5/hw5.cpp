#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <hip/hip_runtime.h>

namespace param {
constexpr int n_steps = 200000;
constexpr double dt = 60;
constexpr double eps = 1e-3;
constexpr double G = 6.674e-11;
constexpr double planet_radius = 1e7;
constexpr double missile_speed = 1e6;
constexpr double eps_sq = eps * eps;
}

#define hipCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true) {
   if (code != hipSuccess) {
      fprintf(stderr,"HIPassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

constexpr int BATCH_SIZE = 200;
constexpr int BLOCK_SIZE = 256;

__global__ void p1_kernel(
    int n, int batch_slot,
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ m,
    const int* __restrict__ device_flags,
    int planet, int asteroid,
    double* __restrict__ batch_out)
{
    __shared__ double s_ax[BLOCK_SIZE];
    __shared__ double s_ay[BLOCK_SIZE];
    __shared__ double s_az[BLOCK_SIZE];

    const int obj = blockIdx.x;
    const int tid = threadIdx.x;
    if (obj >= n) return;

    const double xi = qx_in[obj];
    const double yi = qy_in[obj];
    const double zi = qz_in[obj];

    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = tid; j < n; j += blockDim.x) {
        if (j == obj) continue;
        const bool is_device = device_flags ? device_flags[j] != 0 : false;
        const double mj = is_device ? 0.0 : m[j];
        const double dx = qx_in[j] - xi;
        const double dy = qy_in[j] - yi;
        const double dz = qz_in[j] - zi;
        const double dist = dx * dx + dy * dy + dz * dz + param::eps_sq;
        const double inv = rsqrt(dist);
        const double inv3 = inv * inv * inv;
        ax += param::G * mj * inv3 * dx;
        ay += param::G * mj * inv3 * dy;
        az += param::G * mj * inv3 * dz;
    }

    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_ax[tid] += s_ax[tid + offset];
            s_ay[tid] += s_ay[tid + offset];
            s_az[tid] += s_az[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double nvx = vx_in[obj] + s_ax[0] * param::dt;
        const double nvy = vy_in[obj] + s_ay[0] * param::dt;
        const double nvz = vz_in[obj] + s_az[0] * param::dt;
        const double nqx = xi + nvx * param::dt;
        const double nqy = yi + nvy * param::dt;
        const double nqz = zi + nvz * param::dt;
        qx_out[obj] = nqx; qy_out[obj] = nqy; qz_out[obj] = nqz;
        vx_out[obj] = nvx; vy_out[obj] = nvy; vz_out[obj] = nvz;

        const int offset = batch_slot * 6;
        if (obj == planet) {
            batch_out[offset + 0] = nqx;
            batch_out[offset + 1] = nqy;
            batch_out[offset + 2] = nqz;
        }
        if (obj == asteroid) {
            batch_out[offset + 3] = nqx;
            batch_out[offset + 4] = nqy;
            batch_out[offset + 5] = nqz;
        }
    }
}

__global__ void p2_kernel(
    int n, int batch_slot, double sin_val,
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ m,
    const int* __restrict__ device_flags,
    int planet, int asteroid,
    double* __restrict__ batch_out)
{
    __shared__ double s_ax[BLOCK_SIZE];
    __shared__ double s_ay[BLOCK_SIZE];
    __shared__ double s_az[BLOCK_SIZE];

    const int obj = blockIdx.x;
    const int tid = threadIdx.x;
    if (obj >= n) return;

    const double xi = qx_in[obj];
    const double yi = qy_in[obj];
    const double zi = qz_in[obj];
    const double device_scale = 1.0 + 0.5 * sin_val;

    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = tid; j < n; j += blockDim.x) {
        if (j == obj) continue;
        const bool is_device = device_flags ? device_flags[j] != 0 : false;
        double mj = m[j];
        if (is_device) mj *= device_scale;

        const double dx = qx_in[j] - xi;
        const double dy = qy_in[j] - yi;
        const double dz = qz_in[j] - zi;
        const double dist = dx * dx + dy * dy + dz * dz + param::eps_sq;
        const double inv = rsqrt(dist);
        const double inv3 = inv * inv * inv;
        ax += param::G * mj * inv3 * dx;
        ay += param::G * mj * inv3 * dy;
        az += param::G * mj * inv3 * dz;
    }

    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_ax[tid] += s_ax[tid + offset];
            s_ay[tid] += s_ay[tid + offset];
            s_az[tid] += s_az[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double nvx = vx_in[obj] + s_ax[0] * param::dt;
        const double nvy = vy_in[obj] + s_ay[0] * param::dt;
        const double nvz = vz_in[obj] + s_az[0] * param::dt;
        const double nqx = xi + nvx * param::dt;
        const double nqy = yi + nvy * param::dt;
        const double nqz = zi + nvz * param::dt;
        qx_out[obj] = nqx; qy_out[obj] = nqy; qz_out[obj] = nqz;
        vx_out[obj] = nvx; vy_out[obj] = nvy; vz_out[obj] = nvz;

        const int offset = batch_slot * 6;
        if (obj == planet) {
            batch_out[offset + 0] = nqx;
            batch_out[offset + 1] = nqy;
            batch_out[offset + 2] = nqz;
        }
        if (obj == asteroid) {
            batch_out[offset + 3] = nqx;
            batch_out[offset + 4] = nqy;
            batch_out[offset + 5] = nqz;
        }
    }
}

__global__ void p3_kernel(
    int n, int batch_slot, double sin_val,
    int planet, int asteroid, int target_device, int device_dead,
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out,
    const double* __restrict__ m,
    const int* __restrict__ device_flags,
    double* __restrict__ batch_out)
{
    __shared__ double s_ax[BLOCK_SIZE];
    __shared__ double s_ay[BLOCK_SIZE];
    __shared__ double s_az[BLOCK_SIZE];

    const int obj = blockIdx.x;
    const int tid = threadIdx.x;
    if (obj >= n) return;

    const double xi = qx_in[obj];
    const double yi = qy_in[obj];
    const double zi = qz_in[obj];
    const double device_scale = 1.0 + 0.5 * sin_val;

    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = tid; j < n; j += blockDim.x) {
        if (j == obj) continue;
        const bool is_device = device_flags ? device_flags[j] != 0 : false;
        double mj = m[j];
        if (is_device) {
            if (device_dead && j == target_device) {
                mj = 0.0;
            } else {
                mj *= device_scale;
            }
        }

        const double dx = qx_in[j] - xi;
        const double dy = qy_in[j] - yi;
        const double dz = qz_in[j] - zi;
        const double dist = dx * dx + dy * dy + dz * dz + param::eps_sq;
        const double inv = rsqrt(dist);
        const double inv3 = inv * inv * inv;
        ax += param::G * mj * inv3 * dx;
        ay += param::G * mj * inv3 * dy;
        az += param::G * mj * inv3 * dz;
    }

    s_ax[tid] = ax; s_ay[tid] = ay; s_az[tid] = az;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_ax[tid] += s_ax[tid + offset];
            s_ay[tid] += s_ay[tid + offset];
            s_az[tid] += s_az[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double nvx = vx_in[obj] + s_ax[0] * param::dt;
        const double nvy = vy_in[obj] + s_ay[0] * param::dt;
        const double nvz = vz_in[obj] + s_az[0] * param::dt;
        const double nqx = xi + nvx * param::dt;
        const double nqy = yi + nvy * param::dt;
        const double nqz = zi + nvz * param::dt;
        qx_out[obj] = nqx; qy_out[obj] = nqy; qz_out[obj] = nqz;
        vx_out[obj] = nvx; vy_out[obj] = nvy; vz_out[obj] = nvz;

        const int offset = batch_slot * 9;
        if (obj == planet) {
            batch_out[offset + 0] = nqx;
            batch_out[offset + 1] = nqy;
            batch_out[offset + 2] = nqz;
        }
        if (obj == asteroid) {
            batch_out[offset + 3] = nqx;
            batch_out[offset + 4] = nqy;
            batch_out[offset + 5] = nqz;
        }
        if (obj == target_device) {
            batch_out[offset + 6] = nqx;
            batch_out[offset + 7] = nqy;
            batch_out[offset + 8] = nqz;
        }
    }
}

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n);
    vx.resize(n); vy.resize(n); vz.resize(n);
    m.resize(n); type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) 
         << min_dist << '\n' << hit_time_step << '\n' << gravity_device_id << ' ' << missile_cost << '\n';
}

std::vector<double> precompute_sin() {
    std::vector<double> table(param::n_steps + 1);
    for (int step = 0; step <= param::n_steps; step++) {
        double t = step * param::dt;
        table[step] = fabs(sin(t / 6000.0));
    }
    return table;
}

// P1 simulation - computes min_dist
double simulate_p1(
    int gpu_id,
    int n, int planet, int asteroid,
    const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
    const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
    const std::vector<double>& h_m,
    const std::vector<int>& device_flags_host)
{
    hipCheckError(hipSetDevice(gpu_id));

    double min_dist = 1e100;
    size_t body_bytes = static_cast<size_t>(n) * sizeof(double);

    double *qx[2], *qy[2], *qz[2], *vx[2], *vy[2], *vz[2];
    double *d_m = nullptr;
    double *d_batch_out = nullptr;
    double *h_batch_out = nullptr;

    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMalloc(&qx[i], body_bytes));
        hipCheckError(hipMalloc(&qy[i], body_bytes));
        hipCheckError(hipMalloc(&qz[i], body_bytes));
        hipCheckError(hipMalloc(&vx[i], body_bytes));
        hipCheckError(hipMalloc(&vy[i], body_bytes));
        hipCheckError(hipMalloc(&vz[i], body_bytes));
    }

    hipCheckError(hipMalloc(&d_m, body_bytes));
    hipCheckError(hipMemcpy(d_m, h_m.data(), body_bytes, hipMemcpyHostToDevice));

    int *d_device_flags = nullptr;
    hipCheckError(hipMalloc(&d_device_flags, std::max(1, n) * sizeof(int)));
    hipCheckError(hipMemcpy(d_device_flags, device_flags_host.data(), n * sizeof(int), hipMemcpyHostToDevice));

    hipCheckError(hipMalloc(&d_batch_out, BATCH_SIZE * 6 * sizeof(double)));
    hipCheckError(hipHostMalloc(&h_batch_out, BATCH_SIZE * 6 * sizeof(double)));

    hipStream_t stream;
    hipCheckError(hipStreamCreate(&stream));

    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMemcpyAsync(qx[i], h_qx.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(qy[i], h_qy.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(qz[i], h_qz.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vx[i], h_vx.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vy[i], h_vy.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vz[i], h_vz.data(), body_bytes, hipMemcpyHostToDevice, stream));
    }
    hipCheckError(hipStreamSynchronize(stream));

    double dx0 = h_qx[planet] - h_qx[asteroid];
    double dy0 = h_qy[planet] - h_qy[asteroid];
    double dz0 = h_qz[planet] - h_qz[asteroid];
    min_dist = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);

    int ping = 0, pong = 1;
    int batch_idx = 0;

    for (int step = 1; step <= param::n_steps; step++) {
        hipLaunchKernelGGL(p1_kernel, dim3(n), dim3(BLOCK_SIZE), 0, stream,
            n, batch_idx,
            qx[ping], qy[ping], qz[ping], vx[ping], vy[ping], vz[ping],
            qx[pong], qy[pong], qz[pong], vx[pong], vy[pong], vz[pong],
            d_m, d_device_flags, planet, asteroid, d_batch_out);

        std::swap(ping, pong);
        batch_idx++;

        if (batch_idx >= BATCH_SIZE || step == param::n_steps) {
            hipCheckError(hipMemcpyAsync(h_batch_out, d_batch_out, batch_idx * 6 * sizeof(double), hipMemcpyDeviceToHost, stream));
            hipCheckError(hipStreamSynchronize(stream));

            for (int b = 0; b < batch_idx; b++) {
                const int offset = b * 6;
                double dx = h_batch_out[offset + 0] - h_batch_out[offset + 3];
                double dy = h_batch_out[offset + 1] - h_batch_out[offset + 4];
                double dz = h_batch_out[offset + 2] - h_batch_out[offset + 5];
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < min_dist) min_dist = dist;
            }
            batch_idx = 0;
        }
    }

    hipCheckError(hipHostFree(h_batch_out));
    hipCheckError(hipStreamDestroy(stream));
    hipCheckError(hipFree(d_batch_out));
    hipCheckError(hipFree(d_m));
    hipCheckError(hipFree(d_device_flags));
    for (int i = 0; i < 2; i++) {
        hipCheckError(hipFree(qx[i])); hipCheckError(hipFree(qy[i])); hipCheckError(hipFree(qz[i]));
        hipCheckError(hipFree(vx[i])); hipCheckError(hipFree(vy[i])); hipCheckError(hipFree(vz[i]));
    }

    return min_dist;
}

// P2 simulation - computes hit_step and saves checkpoint
struct P2Result {
    int hit_step = -2;
    std::vector<double> qx, qy, qz, vx, vy, vz;
};

P2Result simulate_p2(
    int gpu_id,
    int n, int planet, int asteroid,
    const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
    const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
    const std::vector<double>& h_m,
    const std::vector<double>& sin_table,
    const std::vector<int>& device_flags_host)
{
    hipCheckError(hipSetDevice(gpu_id));

    P2Result result;
    result.qx = h_qx; result.qy = h_qy; result.qz = h_qz;
    result.vx = h_vx; result.vy = h_vy; result.vz = h_vz;

    size_t body_bytes = static_cast<size_t>(n) * sizeof(double);

    double *qx[2], *qy[2], *qz[2], *vx[2], *vy[2], *vz[2];
    double *d_m = nullptr;
    int *d_device_flags = nullptr;
    double *d_batch_out = nullptr;
    double *h_batch_out = nullptr;

    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMalloc(&qx[i], body_bytes));
        hipCheckError(hipMalloc(&qy[i], body_bytes));
        hipCheckError(hipMalloc(&qz[i], body_bytes));
        hipCheckError(hipMalloc(&vx[i], body_bytes));
        hipCheckError(hipMalloc(&vy[i], body_bytes));
        hipCheckError(hipMalloc(&vz[i], body_bytes));
    }

    hipCheckError(hipMalloc(&d_m, body_bytes));
    hipCheckError(hipMemcpy(d_m, h_m.data(), body_bytes, hipMemcpyHostToDevice));

    hipCheckError(hipMalloc(&d_device_flags, std::max(1, n) * sizeof(int)));
    hipCheckError(hipMemcpy(d_device_flags, device_flags_host.data(), n * sizeof(int), hipMemcpyHostToDevice));

    hipCheckError(hipMalloc(&d_batch_out, BATCH_SIZE * 6 * sizeof(double)));
    hipCheckError(hipHostMalloc(&h_batch_out, BATCH_SIZE * 6 * sizeof(double)));

    hipStream_t stream;
    hipCheckError(hipStreamCreate(&stream));

    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMemcpyAsync(qx[i], h_qx.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(qy[i], h_qy.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(qz[i], h_qz.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vx[i], h_vx.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vy[i], h_vy.data(), body_bytes, hipMemcpyHostToDevice, stream));
        hipCheckError(hipMemcpyAsync(vz[i], h_vz.data(), body_bytes, hipMemcpyHostToDevice, stream));
    }
    hipCheckError(hipStreamSynchronize(stream));

    int ping = 0, pong = 1;
    int batch_idx = 0;
    int batch_start_step = 1;

    for (int step = 1; step <= param::n_steps; step++) {
        double sin_val = sin_table[step];
        hipLaunchKernelGGL(p2_kernel, dim3(n), dim3(BLOCK_SIZE), 0, stream,
            n, batch_idx, sin_val,
            qx[ping], qy[ping], qz[ping], vx[ping], vy[ping], vz[ping],
            qx[pong], qy[pong], qz[pong], vx[pong], vy[pong], vz[pong],
            d_m, d_device_flags, planet, asteroid, d_batch_out);

        std::swap(ping, pong);
        batch_idx++;

        if (batch_idx >= BATCH_SIZE || step == param::n_steps) {
            hipCheckError(hipMemcpyAsync(h_batch_out, d_batch_out, batch_idx * 6 * sizeof(double), hipMemcpyDeviceToHost, stream));
            hipCheckError(hipStreamSynchronize(stream));

            for (int b = 0; b < batch_idx && result.hit_step == -2; b++) {
                const int offset = b * 6;
                double dx = h_batch_out[offset + 0] - h_batch_out[offset + 3];
                double dy = h_batch_out[offset + 1] - h_batch_out[offset + 4];
                double dz = h_batch_out[offset + 2] - h_batch_out[offset + 5];
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (dist < param::planet_radius) {
                    result.hit_step = batch_start_step + b;
                }
            }

            batch_start_step = step + 1;
            batch_idx = 0;
            
            if (result.hit_step != -2) break;
        }
    }

    hipCheckError(hipHostFree(h_batch_out));
    hipCheckError(hipStreamDestroy(stream));
    hipCheckError(hipFree(d_batch_out));
    hipCheckError(hipFree(d_m));
    hipCheckError(hipFree(d_device_flags));
    for (int i = 0; i < 2; i++) {
        hipCheckError(hipFree(qx[i])); hipCheckError(hipFree(qy[i])); hipCheckError(hipFree(qz[i]));
        hipCheckError(hipFree(vx[i])); hipCheckError(hipFree(vy[i])); hipCheckError(hipFree(vz[i]));
    }

    return result;
}

// P3 Workspace (reusable GPU buffers)
struct P3Workspace {
    int gpu_id = 0;
    double *qx[2]{}, *qy[2]{}, *qz[2]{}, *vx[2]{}, *vy[2]{}, *vz[2]{};
    double *d_m = nullptr;
    int *d_device_flags = nullptr;
    double *d_batch_out = nullptr;
    double *h_batch_out = nullptr;
    hipStream_t stream = nullptr;
};

P3Workspace make_p3_workspace(int gpu_id, int n,
    const std::vector<double>& h_m, const std::vector<int>& device_flags_host)
{
    hipCheckError(hipSetDevice(gpu_id));
    P3Workspace ws;
    ws.gpu_id = gpu_id;
    size_t body_bytes = static_cast<size_t>(n) * sizeof(double);
    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMalloc(&ws.qx[i], body_bytes));
        hipCheckError(hipMalloc(&ws.qy[i], body_bytes));
        hipCheckError(hipMalloc(&ws.qz[i], body_bytes));
        hipCheckError(hipMalloc(&ws.vx[i], body_bytes));
        hipCheckError(hipMalloc(&ws.vy[i], body_bytes));
        hipCheckError(hipMalloc(&ws.vz[i], body_bytes));
    }

    hipCheckError(hipMalloc(&ws.d_m, body_bytes));
    hipCheckError(hipMemcpy(ws.d_m, h_m.data(), body_bytes, hipMemcpyHostToDevice));

    hipCheckError(hipMalloc(&ws.d_device_flags, std::max(1, n) * sizeof(int)));
    hipCheckError(hipMemcpy(ws.d_device_flags, device_flags_host.data(), n * sizeof(int), hipMemcpyHostToDevice));

    hipCheckError(hipMalloc(&ws.d_batch_out, BATCH_SIZE * 9 * sizeof(double)));
    hipCheckError(hipHostMalloc(&ws.h_batch_out, BATCH_SIZE * 9 * sizeof(double)));

    hipCheckError(hipStreamCreate(&ws.stream));
    return ws;
}

void destroy_p3_workspace(P3Workspace& ws) {
    hipCheckError(hipSetDevice(ws.gpu_id));
    hipCheckError(hipHostFree(ws.h_batch_out));
    hipCheckError(hipFree(ws.d_batch_out));
    hipCheckError(hipFree(ws.d_m));
    hipCheckError(hipFree(ws.d_device_flags));
    for (int i = 0; i < 2; i++) {
        hipCheckError(hipFree(ws.qx[i])); hipCheckError(hipFree(ws.qy[i])); hipCheckError(hipFree(ws.qz[i]));
        hipCheckError(hipFree(ws.vx[i])); hipCheckError(hipFree(ws.vy[i])); hipCheckError(hipFree(ws.vz[i]));
    }
    hipCheckError(hipStreamDestroy(ws.stream));
}

void load_checkpoint(P3Workspace& ws, int n,
    const std::vector<double>& cp_qx, const std::vector<double>& cp_qy, const std::vector<double>& cp_qz,
    const std::vector<double>& cp_vx, const std::vector<double>& cp_vy, const std::vector<double>& cp_vz)
{
    hipCheckError(hipSetDevice(ws.gpu_id));
    size_t body_bytes = static_cast<size_t>(n) * sizeof(double);
    for (int i = 0; i < 2; i++) {
        hipCheckError(hipMemcpyAsync(ws.qx[i], cp_qx.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
        hipCheckError(hipMemcpyAsync(ws.qy[i], cp_qy.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
        hipCheckError(hipMemcpyAsync(ws.qz[i], cp_qz.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
        hipCheckError(hipMemcpyAsync(ws.vx[i], cp_vx.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
        hipCheckError(hipMemcpyAsync(ws.vy[i], cp_vy.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
        hipCheckError(hipMemcpyAsync(ws.vz[i], cp_vz.data(), body_bytes, hipMemcpyHostToDevice, ws.stream));
    }
    hipCheckError(hipStreamSynchronize(ws.stream));
}

std::pair<int, int> simulate_p3_with_workspace(
    P3Workspace& ws,
    int n, int planet, int asteroid,
    const std::vector<double>& sin_table,
    int target_device)
{
    hipCheckError(hipSetDevice(ws.gpu_id));
    int ping = 0, pong = 1;
    int batch_idx = 0;
    int batch_start_step = 1;
    int hit_step = -2;
    int missile_hit_step = -1;
    bool device_dead = false;

    for (int step = 1; step <= param::n_steps; step++) {
        const double sin_val = sin_table[step];
        hipLaunchKernelGGL(p3_kernel, dim3(n), dim3(BLOCK_SIZE), 0, ws.stream,
            n, batch_idx, sin_val,
            planet, asteroid, target_device, device_dead ? 1 : 0,
            ws.qx[ping], ws.qy[ping], ws.qz[ping], ws.vx[ping], ws.vy[ping], ws.vz[ping],
            ws.qx[pong], ws.qy[pong], ws.qz[pong], ws.vx[pong], ws.vy[pong], ws.vz[pong],
            ws.d_m, ws.d_device_flags, ws.d_batch_out);

        std::swap(ping, pong);
        batch_idx++;

        if (batch_idx >= BATCH_SIZE || step == param::n_steps) {
            hipCheckError(hipMemcpyAsync(ws.h_batch_out, ws.d_batch_out, batch_idx * 9 * sizeof(double), hipMemcpyDeviceToHost, ws.stream));
            hipCheckError(hipStreamSynchronize(ws.stream));

            for (int b = 0; b < batch_idx && hit_step == -2; b++) {
                const int current_step = batch_start_step + b;
                const int offset = b * 9;

                if (!device_dead) {
                    double dx = ws.h_batch_out[offset + 0] - ws.h_batch_out[offset + 6];
                    double dy = ws.h_batch_out[offset + 1] - ws.h_batch_out[offset + 7];
                    double dz = ws.h_batch_out[offset + 2] - ws.h_batch_out[offset + 8];
                    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                    double missile_dist = static_cast<double>(current_step) * param::dt * param::missile_speed;
                    if (missile_dist >= dist) {
                        device_dead = true;
                        missile_hit_step = current_step;
                    }
                }

                double dx = ws.h_batch_out[offset + 0] - ws.h_batch_out[offset + 3];
                double dy = ws.h_batch_out[offset + 1] - ws.h_batch_out[offset + 4];
                double dz = ws.h_batch_out[offset + 2] - ws.h_batch_out[offset + 5];
                if (std::sqrt(dx * dx + dy * dy + dz * dz) < param::planet_radius) {
                    hit_step = current_step;
                }
            }

            batch_start_step = step + 1;
            batch_idx = 0;
        }
        if (hit_step != -2) break;
    }

    return {hit_step, missile_hit_step};
}

// Worker thread for dynamic task dispatch
void worker_thread_dynamic(
    int gpu_id,
    std::atomic<int>& task_counter,
    const std::vector<int>& device_ids,
    std::vector<std::pair<int, int>>& p3_results,
    int n, int planet, int asteroid,
    const std::vector<double>& cp_qx, const std::vector<double>& cp_qy, const std::vector<double>& cp_qz,
    const std::vector<double>& cp_vx, const std::vector<double>& cp_vy, const std::vector<double>& cp_vz,
    const std::vector<double>& h_m,
    const std::vector<double>& sin_table,
    const std::vector<int>& device_flags)
{
    hipCheckError(hipSetDevice(gpu_id));
    
    // Create workspace for this GPU
    P3Workspace ws = make_p3_workspace(gpu_id, n, h_m, device_flags);

    while (true) {
        // Atomically get next task
        int task_idx = task_counter.fetch_add(1);
        
        if (task_idx >= static_cast<int>(device_ids.size())) {
            break; // No more tasks
        }

        // Load checkpoint and run simulation for this target device
        load_checkpoint(ws, n, cp_qx, cp_qy, cp_qz, cp_vx, cp_vy, cp_vz);
        p3_results[task_idx] = simulate_p3_with_workspace(ws, n, planet, asteroid, sin_table, device_ids[task_idx]);
    }

    destroy_p3_workspace(ws);
}

int main(int argc, char** argv) {
    if (argc != 3) throw std::runtime_error("must supply 2 arguments");

    int deviceCount = 0;
    hipCheckError(hipGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        throw std::runtime_error("Need at least 1 GPU");
    }

    int n, planet, asteroid;
    std::vector<double> h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m;
    std::vector<std::string> h_type;

    read_input(argv[1], n, planet, asteroid, h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, h_type);

    std::vector<int> device_ids;
    std::vector<int> device_flags(n, 0);
    for (int i = 0; i < n; i++) {
        if (h_type[i] == "device") {
            device_ids.push_back(i);
            device_flags[i] = 1;
        }
    }

    std::vector<double> sin_table = precompute_sin();
    
    // Run P1 and P2 in parallel on different GPUs
    double p1_min_dist = 0.0;
    P2Result p2_result;
    
    if (deviceCount >= 2) {
        std::thread t0([&]() { 
            p1_min_dist = simulate_p1(0, n, planet, asteroid, 
                h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, device_flags); 
        });
        std::thread t1([&]() { 
            p2_result = simulate_p2(1, n, planet, asteroid, 
                h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, sin_table, device_flags); 
        });
        t0.join(); t1.join();
    } else {
        p1_min_dist = simulate_p1(0, n, planet, asteroid, 
            h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, device_flags);
        p2_result = simulate_p2(0, n, planet, asteroid, 
            h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, sin_table, device_flags);
    }

    int best_device_id = -1;
    double min_cost = 1e100;

    if (p2_result.hit_step != -2 && !device_ids.empty()) {
        const int num_devices_to_check = static_cast<int>(device_ids.size());
        std::vector<std::pair<int, int>> p3_results(num_devices_to_check);

        std::atomic<int> task_counter(0);
        
        std::vector<std::thread> workers;
        for (int g = 0; g < deviceCount; g++) {
            workers.emplace_back(worker_thread_dynamic,
                g,
                std::ref(task_counter),
                std::ref(device_ids),
                std::ref(p3_results),
                n, planet, asteroid,
                std::ref(p2_result.qx), std::ref(p2_result.qy), std::ref(p2_result.qz),
                std::ref(p2_result.vx), std::ref(p2_result.vy), std::ref(p2_result.vz),
                std::ref(h_m),
                std::ref(sin_table),
                std::ref(device_flags));
        }

        for (auto& t : workers) {
            t.join();
        }

        for (int i = 0; i < num_devices_to_check; i++) {
            auto [hit_step, missile_step] = p3_results[i];
            if (missile_step != -1 && hit_step == -2) {
                double cost = 1e5 + 1e3 * (missile_step + 1) * param::dt;
                if (cost < min_cost) {
                    min_cost = cost;
                    best_device_id = device_ids[i];
                }
            }
        }
    }

    if (best_device_id == -1) {
        write_output(argv[2], p1_min_dist, p2_result.hit_step, -1, 0);
    } else {
        write_output(argv[2], p1_min_dist, p2_result.hit_step, best_device_id, min_cost);
    }
    
    return 0;
}
