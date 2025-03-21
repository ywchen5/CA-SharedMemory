#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>


// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)



/**
 * 不使用共享内存的矩阵乘法
 * 每个线程计算输出矩阵的一个元素
 */


__global__ void matrixMulWithoutShared(float *A, float *B, float *C, int width) {
    // 计算当前线程对应的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

/**
 * 使用共享内存的矩阵乘法
 * 利用共享内存提高内存访问效率
 */
__global__ void matrixMulWithShared(float *A, float *B, float *C, int width) {
    const int TILE_WIDTH = 16;
    
    // 为输入矩阵的块分配共享内存
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    
    // 计算全局索引
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 计算当前线程的行和列索引
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    // 遍历所有的块
    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // 加载数据到共享内存中
        if (row < width && t * TILE_WIDTH + tx < width)
            sharedA[ty][tx] = A[row * width + t * TILE_WIDTH + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if (col < width && t * TILE_WIDTH + ty < width)
            sharedB[ty][tx] = B[(t * TILE_WIDTH + ty) * width + col];
        else
            sharedB[ty][tx] = 0.0f;
        
        // 同步以确保块已经加载完成
        __syncthreads();
        
        // 计算当前块的部分结果
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        // 同步后再加载下一块
        __syncthreads();
    }
    
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

/**
 * 不使用共享内存的向量加法
 * 作为对比基准
 */
__global__ void vectorAddWithoutShared(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 使用共享内存的向量加法
 * 演示共享内存可能不会带来明显优势的情况
 */
__global__ void vectorAddWithShared(float *A, float *B, float *C, int n) {
    const int BLOCK_SIZE = 256;
    __shared__ float sharedA[BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        sharedA[tid] = A[idx];
        sharedB[tid] = B[idx];
    }
    
    __syncthreads();
    
    if (idx < n) {
        C[idx] = sharedA[tid] + sharedB[tid];
    }
}

/**
 * 不使用共享内存的归约求和
 * 使用原子操作直接累加到全局内存
 */
__global__ void reductionSumWithoutShared(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

/**
 * 使用共享内存的归约求和
 * 先在共享内存中进行部分归约，减少全局内存的原子操作次数
 */
__global__ void reductionSumWithShared(float *input, float *output, int n) {
    const int BLOCK_SIZE = 256;
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // 在共享内存中进行归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 将结果写回全局内存
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * 初始化矩阵/向量数据
 */
void initializeData(std::vector<float> &data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (auto &val : data) {
        val = dist(gen);
    }
}

/**
 * CPU版本矩阵乘法用于验证结果
 */
void cpuMatrixMul(const std::vector<float> &A, const std::vector<float> &B, 
                  std::vector<float> &C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

/**
 * CPU版本向量加法用于验证结果
 */
void cpuVectorAdd(const std::vector<float> &A, const std::vector<float> &B, 
                 std::vector<float> &C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

/**
 * CPU版本求和用于验证结果
 */
float cpuReductionSum(const std::vector<float> &input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

/**
 * 检查GPU结果与CPU结果是否匹配
 */
bool checkResult(const std::vector<float> &cpuResult, const std::vector<float> &gpuResult, 
                const std::string &testName) {
    const float epsilon = 1e-5;
    for (size_t i = 0; i < cpuResult.size(); i++) {
        if (std::abs(cpuResult[i] - gpuResult[i]) > epsilon) {
            std::cout << testName << ": 结果不匹配! 索引 " << i 
                      << ", CPU: " << cpuResult[i] 
                      << ", GPU: " << gpuResult[i] << std::endl;
            return false;
        }
    }
    std::cout << testName << ": 结果匹配!" << std::endl;
    return true;
}

int main() {
    // 测试不同矩阵大小
    const std::vector<int> matrixSizes = {128, 512, 1024, 2048};
    
    // 设置CUDA事件来测量时间
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    std::cout << "======= 矩阵乘法性能测试 =======" << std::endl;
    
    for (int width : matrixSizes) {
        std::cout << "\n矩阵大小: " << width << "x" << width << std::endl;
        
        size_t matrixBytes = width * width * sizeof(float);
        
        // 分配和初始化主机内存
        std::vector<float> h_A(width * width);
        std::vector<float> h_B(width * width);
        std::vector<float> h_C_cpu(width * width);
        std::vector<float> h_C_gpu_no_shared(width * width);
        std::vector<float> h_C_gpu_shared(width * width);
        
        initializeData(h_A);
        initializeData(h_B);

        // 分配设备内存
        float *d_A, *d_B, *d_C_no_shared, *d_C_shared;
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, matrixBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, matrixBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_no_shared, matrixBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_shared, matrixBytes));

        // 复制数据到设备
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), matrixBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), matrixBytes, cudaMemcpyHostToDevice));

        // 设置网格和块大小
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                      (width + blockSize.y - 1) / blockSize.y);
        
        // 不使用共享内存的矩阵乘法
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        matrixMulWithoutShared<<<gridSize, blockSize>>>(d_A, d_B, d_C_no_shared, width);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float timeNoShared = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeNoShared, start, stop));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_no_shared.data(), d_C_no_shared, matrixBytes, cudaMemcpyDeviceToHost));
        
        // 使用共享内存的矩阵乘法
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        matrixMulWithShared<<<gridSize, blockSize>>>(d_A, d_B, d_C_shared, width);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float timeShared = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeShared, start, stop));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_shared.data(), d_C_shared, matrixBytes, cudaMemcpyDeviceToHost));

        // 计算CPU参考结果
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuMatrixMul(h_A, h_B, h_C_cpu, width);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpuTime = cpuEnd - cpuStart;
        
        // 检查和比较结果
        std::cout << "CPU 时间: " << cpuTime.count() << " ms" << std::endl;
        std::cout << "GPU 不使用共享内存: " << timeNoShared << " ms" << std::endl;
        std::cout << "GPU 使用共享内存: " << timeShared << " ms" << std::endl;
        std::cout << "加速比 (不使用/使用共享内存): " << timeNoShared / timeShared << "x" << std::endl;
        
        checkResult(h_C_cpu, h_C_gpu_no_shared, "矩阵乘法 (不使用共享内存)");
        checkResult(h_C_cpu, h_C_gpu_shared, "矩阵乘法 (使用共享内存)");
        
        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_no_shared);
        cudaFree(d_C_shared);
    }
    
    std::cout << "\n======= 向量加法性能测试 =======" << std::endl;
    
    // 向量加法测试 (展示共享内存可能不会明显提高性能的情况)
    for (int size : {1000000, 10000000}) {
        std::cout << "\n向量大小: " << size << std::endl;
        
        size_t vectorBytes = size * sizeof(float);
        
        // 分配和初始化主机内存
        std::vector<float> h_A(size);
        std::vector<float> h_B(size);
        std::vector<float> h_C_cpu(size);
        std::vector<float> h_C_gpu_no_shared(size);
        std::vector<float> h_C_gpu_shared(size);
        
        initializeData(h_A);
        initializeData(h_B);

        // 分配设备内存
        float *d_A, *d_B, *d_C_no_shared, *d_C_shared;
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, vectorBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, vectorBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_no_shared, vectorBytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C_shared, vectorBytes));

        // 复制数据到设备
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), vectorBytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), vectorBytes, cudaMemcpyHostToDevice));

        // 设置网格和块大小
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        // 不使用共享内存的向量加法
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        vectorAddWithoutShared<<<gridSize, blockSize>>>(d_A, d_B, d_C_no_shared, size);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float timeNoShared = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeNoShared, start, stop));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_no_shared.data(), d_C_no_shared, vectorBytes, cudaMemcpyDeviceToHost));
        
        // 使用共享内存的向量加法
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        vectorAddWithShared<<<gridSize, blockSize>>>(d_A, d_B, d_C_shared, size);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float timeShared = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeShared, start, stop));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_shared.data(), d_C_shared, vectorBytes, cudaMemcpyDeviceToHost));
        
        // 计算CPU参考结果
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuVectorAdd(h_A, h_B, h_C_cpu, size);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpuTime = cpuEnd - cpuStart;
        
        // 检查和比较结果
        std::cout << "CPU 时间: " << cpuTime.count() << " ms" << std::endl;
        std::cout << "GPU 不使用共享内存: " << timeNoShared << " ms" << std::endl;
        std::cout << "GPU 使用共享内存: " << timeShared << " ms" << std::endl;
        std::cout << "性能比例 (不使用/使用共享内存): " << timeNoShared / timeShared << "x" << std::endl;
        
        checkResult(h_C_cpu, h_C_gpu_no_shared, "向量加法 (不使用共享内存)");
        checkResult(h_C_cpu, h_C_gpu_shared, "向量加法 (使用共享内存)");
        
        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_no_shared);
        cudaFree(d_C_shared);
    }
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}