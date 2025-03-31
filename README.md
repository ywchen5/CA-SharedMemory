
<div align="center">
  <h1> CUDA共享内存研究 </h1>
  <p>
    <a href="#项目概述">项目概述</a> •
    <a href="#环境要求">环境要求</a> •
    <a href="#项目结构">项目结构</a> •
    <a href="#使用方法">使用方法</a> •
    <a href="#参考资料">参考资料</a>
  </p>
</div>

---
<a id = "项目概述"></a>
## 📋 项目概述
CUDA共享内存是CUDA编程模型中的一种重要特性，它允许线程块中的线程共享数据，从而提高了数据访问的速度和效率。共享内存的使用可以显著减少对全局内存的访问，从而提高程序的性能。

<a id = "环境要求"></a>

## 🔧 环境要求
- CUDA 12.6
- NVIDIA RTX 4060(Ada架构)


<a id = "项目结构"></a>

## 📁 项目结构
```
CA-SharedMemory
├── bin # 可执行文件
├── obj # 中间文件
├── report # 报告
│   └── assets # 图片资源
└── src
    ├── env_test # 环境测试
    └── shared_mem_test # 主要测试
```

<a id = "使用方法"></a>

## 🚀 使用方法

#### 编译
```bash
make 
```
#### 运行
```bash
make run
```
#### 清理
```bash
make clean
```


<a id = "参考资料"></a>

## 📚 参考资料
[1]https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
[2]https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
[3]https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
[4]https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
[5]Konda S, Monarch I, Sargent P, et al. Shared memory in design: A unifying theme for research and practice[J]. Research in Engineering design, 1992, 4: 23-42.
