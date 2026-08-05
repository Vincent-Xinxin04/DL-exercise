# 使用包含 CUDA 运行时的基础镜像以支持 GPU
# 升级至 Ubuntu 24.04 以获取更高版本的 Python (3.12)，满足 pandas 3.0.2 的需求
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 安装 Python 和必要系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgomp1 \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 python3 为默认 python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 将 requirements.txt 复制到容器中
COPY requirements.txt .

# 直接安装依赖，并添加 PyTorch 的 CUDA 索引地址
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 将当前目录内容复制到容器中的 /app
COPY . .

# 删除本地 venv
RUN rm -rf venv

# 容器启动时默认运行预测脚本
CMD ["python", "源码/test.py"]
