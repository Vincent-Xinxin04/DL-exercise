# 1. 基础镜像：使用支持 CUDA 12.6 和 Ubuntu 24.04 的官方镜像
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# 2. 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. 优化系统源并安装 Python 3.12 (Ubuntu 24.04 默认版本)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 4. 配置 pip 使用清华镜像源（比阿里云更稳定，且支持哈希校验）
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.timeout 6000

# 5. 分步安装大型依赖包 (利用 Docker 缓存机制)
# 针对 Ubuntu 24.04，必须使用 --break-system-packages 或者创建 venv
# 先安装 Torch (指向官方 WHL 源以保证 CUDA 版本正确)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126

# 再安装 CatBoost 和 XGBoost (容易断连的大包)
RUN pip3 install --no-cache-dir --break-system-packages \
    catboost==1.2.10 xgboost==3.2.0

# 6. 安装剩余的依赖项
COPY requirements.txt .
# 过滤掉已经手动安装过的包，避免重复扫描
RUN grep -vE 'torch|catboost|xgboost' requirements.txt > requirements_light.txt && \
    pip3 install --no-cache-dir --break-system-packages -r requirements_light.txt

# 7. 复制源代码
COPY . .

# 8. 默认运行命令
CMD ["python", "源码/test.py"]
