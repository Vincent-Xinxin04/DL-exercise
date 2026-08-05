# 使用 Python 3.10 官方 Slim 镜像（匹配本地 venv 版本 3.10.20）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
# 防止 Python 产生 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
# 强制 stdout 和 stderr 流不被缓冲
ENV PYTHONUNBUFFERED=1
# 设置 PYTHONPATH 确保可以导入源码（直接定义路径避免 UndefinedVar 警告）
ENV PYTHONPATH="/app/源码"

# 安装系统依赖
# 移除 build-essential 以解决构建时内存不足的问题
# 大多数库（如 capstone）都有预编译的 wheel，不需要在容器内编译
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# 配置 pip 镜像加速（清华源）
# 同时配置 PyTorch 的 extra-index-url
RUN pip install --no-cache-dir -U pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://download.pytorch.org/whl/cu126

# 首先复制 requirements.txt 以利用 Docker 层缓存
COPY requirements.txt .

# --- 分步安装以应对大文件下载 ---
# 1. 先单独安装最重的 Torch 相关包，增加重试次数以防断线
RUN pip install --no-cache-dir --default-timeout=1000 --retries 20 \
    torch==2.11.0+cu126 \
    torchvision==0.26.0+cu126

# 2. 安装其余依赖
RUN pip install --no-cache-dir --default-timeout=1000 --retries 20 -r requirements.txt

# 复制项目所有文件
# .dockerignore 会自动过滤掉 venv, __pycache__, 数据集等不需要的内容
COPY . .

# 确保必要的目录存在
RUN mkdir -p 模型 提交结果

# 默认运行测试脚本
CMD ["python", "源码/test.py"]


