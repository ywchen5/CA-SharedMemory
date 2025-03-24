# CUDA编译器和标志
NVCC := nvcc
NVCC_FLAGS := -O0 # 不使用任何优化

# 根据显卡支持的CUDA计算能力添加目标架构
# 常见架构: 70=Volta, 75=Turing, 80=Ampere, 86=Ampere(RTX 30系列), 89=Ada, 90=Hopper
# 可通过nvidia-smi --query-gpu=compute_cap --format=csv查看您的GPU计算能力
ARCH_FLAGS := -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90

# 项目结构
SRC_DIR := src/shared_mem_test
OBJ_DIR := obj
BIN_DIR := bin
TARGET := $(BIN_DIR)/shared_mem_test.out

# 源文件
SRCS := $(SRC_DIR)/test.cu
OBJS := $(OBJ_DIR)/test.o

# 创建必要的目录
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(BIN_DIR))

# 默认目标
all: $(TARGET)

# 链接目标
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -o $@ $^

# 编译源文件
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c -o $@ $<

# 清理
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

# 运行
run: $(TARGET)
	$(TARGET)

.PHONY: all clean run