"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // TODO: implement warp-level sum reduction using shuffle intrinsics
    #pragma unroll
    for (int i = warpSize / 2; i > 0; i /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // TODO: implement warp-level max reduction using shuffle intrinsics
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // TODO: block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < num_warps) ? shared[lane] : 0.0f;
        return warp_reduce_sum(v);
    }
    return 0.0f;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    // TODO: block-wide max via warp_reduce_max + shared memory
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane < num_warps ? shared[lane] : -INFINITY;
        return warp_reduce_max(val);
    }

    return 0.0f;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  (void)x; (void)residual; (void)out; (void)n;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + residual[idx];
  }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = 0.5 * v * (1 + tanhf(sqrtf(2.0f / M_PI) * (v + 0.044715 * v * v * v)));
    }

}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    (void)x; (void)out; (void)n;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        out[idx] = x[idx] / (1 + expf(-x[idx]));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    (void)gate; (void)up; (void)out; (void)n;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        out[idx] = gate[idx] / (1 + expf(-gate[idx])) * up[idx];
    }
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    // TODO: Apply RMSNorm per row (one block per row)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + (size_t)blockIdx.x * n;
    float* out_row = out + (size_t)blockIdx.x * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }
    __shared__ float shared[32];
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float inv;
    if (tid == 0) {
        inv = rsqrtf(sum_sq / n + eps);
    }
    __syncthreads();
    
    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * inv * weight[i];
    }
}

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // TODO: per-row LayerNorm using block_reduce_sum for mean and variance
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + (size_t)n * row;
    float* out_row = out + (size_t)n * row;

    float sum_1 = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum_1 += x_row[i];
        sum_sq += x_row[i] * x_row[i];
    }

    __shared__ float shared1[32], shared2[32];
    sum_1 = block_reduce_sum(sum_1, shared1);
    sum_sq = block_reduce_sum(sum_sq, shared2);

    __shared__ float mean, inv;
    if (tid == 0) {
        mean = sum_1 / n;
        inv = rsqrtf(sum_sq / n - mean * mean + eps);
    }
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // TODO: fuse residual addition with RMSNorm (one block per row)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + (size_t)blockIdx.x * n;
    const float* residual_row = residual + (size_t)blockIdx.x * n;

    float* residual_out_row = residual_out + (size_t)blockIdx.x * n;
    float* out_row = out + (size_t)blockIdx.x * n;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float r_i = x_row[i] + residual_row[i];
        residual_out_row[i] = r_i;
        sum_sq += r_i * r_i;
    }
    __shared__ float shared[32];
    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float inv;
    if (tid == 0) {
        inv = rsqrtf(sum_sq / n + eps);
    }
    __syncthreads();
    
    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = residual_out_row[i] * inv * weight[i];
    }
}

# Step 12 - softmax_row_kernel
__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax (one block per row)
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + (size_t)row * cols;
    float* out_row = out + (size_t)row * cols;

    float max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max = fmaxf(max, x_row[i]);
    }

    __shared__ float shared[32];
    max = block_reduce_max(max, shared);
    __shared__ float row_max;
    if (tid == 0) {
        row_max = max;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += expf(x_row[i] - row_max);
    }

    sum = block_reduce_sum(sum, shared);
    __shared__ float row_sum;
    if (tid == 0) {
        row_sum = sum;
    }
    __syncthreads();


    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}

# Step 13 - causal_softmax_kernel
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: numerically stable causal softmax (one block per row);
    //       mask columns c > row to 0; use block_reduce_max / block_reduce_sum
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x_row = x + (size_t)row * cols;
    float* out_row = out + (size_t)row * cols;

    float max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max = i <= row ? fmaxf(max, x_row[i]) : max;
    }

    __shared__ float shared[32];
    max = block_reduce_max(max, shared);
    __shared__ float row_max;
    if (tid == 0) {
        row_max = max;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += i <= row ? expf(x_row[i] - row_max) : 0.0f;
    }

    sum = block_reduce_sum(sum, shared);
    __shared__ float row_sum;
    if (tid == 0) {
        row_sum = sum;
    }
    __syncthreads();


    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = i <= row ? expf(x_row[i] - row_max) / row_sum : 0.0f;
    }
}

# Step 14 - embedding_lookup_kernel
__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    // TODO: gather embedding vectors for each token id into out
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int row = tid / embed_dim;
    int col = tid % embed_dim;
    if (tid < seq_len * embed_dim) {
        int token_id = token_ids[row];
        const float* w = weight + (size_t)token_id * embed_dim;
        out[tid] = w[col];
    }
}

# Step 15 - rope_kernel
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    // TODO: apply RoPE rotation in-place to every even/odd pair of q and k
    int half = head_dim / 2;
    int total = seq_len * n_heads * half;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < total) {
        int i = tid % half;
        int h = (tid / half) % n_heads;
        int t = (tid / half / n_heads) % seq_len;

        int base = (t * n_heads + h) * head_dim;
        int even = base + 2 * i;
        int odd = even + 1;
        float cos = cos_table[t * half + i];
        float sin = sin_table[t * half + i];

        float q0 = q[even];
        float q1 = q[odd];
        float k0 = k[even];
        float k1 = k[odd];

        q[even] = q0 * cos - q1 * sin;
        q[odd] = q0 * sin + q1 * cos; 
        k[even] = k0 * cos - k1 * sin;
        k[odd] = k0 * sin + k1 * cos; 
    }
}

# Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // TODO: compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;
    int m = tid / N, n = tid % N;
    const float* x_row = x + m * K;
    const float* weight_col = weight + n * K;
    float sum = (bias != nullptr) ? bias[n] : 0.0f;
    for (int i = 0; i < K; i++) {
        sum += x_row[i] * weight_col[i];
    } 

    out[tid] = sum;
}

# Step 17 - fused_linear_bias_gelu_kernel
__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // TODO: fuse matmul, bias add, and GELU tanh approx into one kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * N) return;
    int m = tid / N, n = tid % N;
    const float* x_row = x + m * K;
    const float* weight_col = weight + n * K;
    float v = (bias != nullptr) ? bias[n] : 0.0f;
    for (int i = 0; i < K; i++) {
        v += x_row[i] * weight_col[i];
    } 

    out[tid] = 0.5 * v * (1 + tanhf(sqrtf(2.0f / M_PI) * (v + 0.044715 * v * v * v)));
}

# Step 18 - mlp_swiglu_forward
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // TODO: allocate temps, run gate/up linears, swiglu, then down projection
    (void)x; (void)w_gate; (void)w_up; (void)w_down; (void)out;
    (void)M; (void)hidden_dim; (void)intermediate_dim;

    const int threads = 256;
    float *d_gate, *d_up, *d_act;
    cudaMalloc(&d_gate, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_act, M * intermediate_dim * sizeof(float));
    
    int blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(x, w_gate, nullptr, d_gate, M, intermediate_dim, hidden_dim);
    linear_kernel<<<blocks, threads>>>(x, w_up, nullptr, d_up, M, intermediate_dim, hidden_dim);
    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_act, M * intermediate_dim);
    
    blocks = (M * hidden_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(d_act, w_down, nullptr, out, M, hidden_dim, intermediate_dim);
    cudaDeviceSynchronize();
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_act);
}

# Step 19 - rmsnorm_residual_block
void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    // TODO: launch fused_add_rmsnorm_kernel for the pre-norm residual+RMSNorm block
    const int threads = 256;
    
    int blocks = (rows * n + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads>>>(x, residual, residual_out, rows * n);
    rmsnorm_kernel<<<rows, threads>>>(residual_out, weight, out, n, eps);
}

# Step 20 - run_transformer_ffn
void run_transformer_ffn(const float* x, const float* residual,
                         const float* norm_weight, const float* w_gate,
                         const float* w_up, const float* w_down, float* out,
                         int M, int hidden_dim, int intermediate_dim,
                         float eps) {
  // TODO: residual+RMSNorm, SwiGLU MLP, then residual add into out
  float *residual_out, *block_out;
  int n = M * hidden_dim;
  cudaMalloc(&residual_out, n * sizeof(float));
  cudaMalloc(&block_out, n * sizeof(float));

  rmsnorm_residual_block(x, residual, norm_weight, block_out, residual_out, M, hidden_dim, eps);
  mlp_swiglu_forward(block_out, w_gate, w_up, w_down, out, M, hidden_dim, intermediate_dim);


  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  add_residual_kernel<<<blocks, threads>>>(residual_out, out, out, n);
  cudaDeviceSynchronize();
  cudaFree(residual_out);
  cudaFree(block_out);
}

