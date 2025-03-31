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
 * 不使用共享内存的归约求和
 * 使用原子操作直接累加到全局内存
 */
// __global__ void reductionSumWithoutShared(float *input, float *output, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < n) {
//         atomicAdd(output, input[idx]);
//     }
// }

__global__ void reductionSumWithoutSharedButParallel(float *input, float *output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 不使用共享内存，直接在全局内存上完成归约
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && idx + stride < n) {
            input[idx] += input[idx + stride];
        }
        __syncthreads(); // 块内同步
    }
    
    // 每个块的第一个线程将结果写回
    if (tid == 0) {
        atomicAdd(output, input[blockIdx.x * blockDim.x]);
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
    const float epsilon = 1e-3;
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
    std::cout << "\n======= 规约求和性能测试 =======" << std::endl;
    
    // 规约求和测试（展示共享内存可提高性能的情况）
    for (int size : {65536, 32768, 16777216}) {
        // 设置CUDA事件来测量时间
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        std::cout << "\n向量大小: " << size << std::endl;
        
        size_t vectorBytes = size * sizeof(float);
        
        // 分配和初始化主机内存
        std::vector<float> h_input(size);
        float h_sum_cpu = 0.0f;
        float h_sum_no_shared = 0.0f;
        float h_sum_shared = 0.0f;
        
        initializeData(h_input); // host data initialization
    
        // 分配设备内存
        float *d_input, *d_sum_no_shared, *d_sum_shared; 
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, vectorBytes)); // 分配输入向量内存
        CHECK_CUDA_ERROR(cudaMalloc(&d_sum_shared, sizeof(float))); // 分配使用共享内存的输出内存
        CHECK_CUDA_ERROR(cudaMalloc(&d_sum_no_shared, sizeof(float))); // 分配不使用共享内存的输出内存

        // 复制数据到设备
        
        // 设置网格和块大小
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        
        // 不使用共享内存的规约求和
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), vectorBytes, cudaMemcpyHostToDevice)); // 从主机复制到设备
        CHECK_CUDA_ERROR(cudaMemset(d_sum_no_shared, 0, sizeof(float))); // 清零不使用共享内存的输出
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        // reductionSumWithoutShared<<<gridSize, blockSize>>>(d_input, d_sum_no_shared, size);
        reductionSumWithoutSharedButParallel<<<gridSize, blockSize>>>(d_input, d_sum_no_shared, size);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float timeNoShared = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeNoShared, start, stop));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_no_shared, d_sum_no_shared, sizeof(float), cudaMemcpyDeviceToHost)); // 从设备复制结果到主机


        // 使用共享内存的规约求和
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), vectorBytes, cudaMemcpyHostToDevice)); // 从主机复制到设备
        CHECK_CUDA_ERROR(cudaMemset(d_sum_shared, 0, sizeof(float))); // 清零共享内存
        CHECK_CUDA_ERROR(cudaEventRecord(start)); // 开始计时
        reductionSumWithShared<<<gridSize, blockSize>>>(d_input, d_sum_shared, size); // 调用规约核函数
        CHECK_CUDA_ERROR(cudaEventRecord(stop)); // 停止计时
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // 等待核函数完成
        
        float timeShared = 0; // 存储共享内存的时间
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeShared, start, stop)); // 计算时间
        CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_shared, d_sum_shared, sizeof(float), cudaMemcpyDeviceToHost)); // 从设备复制结果到主机

        // 计算CPU参考结果
        auto cpuStart = std::chrono::high_resolution_clock::now();
        h_sum_cpu = cpuReductionSum(h_input, size);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpuTime = cpuEnd - cpuStart;
        
        // 检查和比较结果
        const float epsilon = 1e-4;
        bool noSharedCorrect = std::abs(h_sum_cpu - h_sum_no_shared) < epsilon * size;
        bool sharedCorrect = std::abs(h_sum_cpu - h_sum_shared) < epsilon * size;
        
        std::cout << "CPU 时间: " << cpuTime.count() << " ms, 结果: " << h_sum_cpu << std::endl;
        std::cout << "GPU 不使用共享内存: " << timeNoShared << " ms, 结果: " << h_sum_no_shared 
                  << (noSharedCorrect ? " (正确)" : " (不正确)") << std::endl;
        std::cout << "GPU 使用共享内存: " << timeShared << " ms, 结果: " << h_sum_shared 
                  << (sharedCorrect ? " (正确)" : " (不正确)") << std::endl;
        std::cout << "加速比 (不使用/使用共享内存): " << timeNoShared / timeShared << "x" << std::endl;
        
        // 释放设备内存
        cudaFree(d_input);
        cudaFree(d_sum_no_shared);
        cudaFree(d_sum_shared);
        // 销毁CUDA事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}