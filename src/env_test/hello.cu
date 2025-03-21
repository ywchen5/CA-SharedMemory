#include <stdio.h>

__global__ void helloFromGPU(void)
{
  printf("Hello World from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
  // 检查CUDA设备
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  
  if (error != cudaSuccess) {
    printf("CUDA错误: %s\n", cudaGetErrorString(error));
    return -1;
  }
  
  if (deviceCount == 0) {
    printf("没有找到支持CUDA的设备\n");
    return -1;
  }
  
  printf("找到%d个CUDA设备\n", deviceCount);
  printf("Hello World from CPU!\n");

  // 启动内核
  helloFromGPU<<<2, 4>>>();
  
  // 检查启动错误
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("内核启动错误: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // 同步并刷新输出
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    printf("同步错误: %s\n", cudaGetErrorString(error));
    return -1;
  }
  
  // 重置设备以确保所有输出都被刷新
  cudaDeviceReset();
  
  return 0;
}