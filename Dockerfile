# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:20.04

# 设置环境变量，避免一些交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 更换为阿里云源
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list

# 更新 apt-get 并安装必要的依赖
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y bzip2
RUN apt-get install -y ca-certificates
RUN apt-get install -y git
RUN apt-get install -y vim

# 清理 apt-get 缓存
RUN apt-get clean

# 将 Anaconda 安装包复制到容器中
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/Anaconda3.sh

#### 下载速度慢的话可以手动下载这几个包，然后手动安装（强烈建议！！！！！！！）
RUN wget https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp311-cp311-linux_x86_64.whl#sha256=16e4ef3b32b45a278a0c512723f81cfa57035ebd5a75dbc2fb1360197ae06acd -O /tmp/torch-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
RUN wget https://download.pytorch.org/whl/cu124/torchaudio-2.4.1%2Bcu124-cp311-cp311-linux_x86_64.whl#sha256=9045eeb86e8fc8767fb75224978051b114724d19af8f566c9dbd2c5c7d150f91 -O /tmp/torchaudio-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
RUN wget https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp311-cp311-linux_x86_64.whl#sha256=a5f7eb5ef22f34a7d18fcbc27b6c01f7dde5cd530df311cdbdd31169f91cbd98 -O /tmp/torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl

# COPY DockerENV/torch-2.4.1+cu124-cp311-cp311-linux_x86_64.whl /tmp/torch-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
# COPY DockerENV/torchaudio-2.4.1+cu124-cp311-cp311-linux_x86_64.whl /tmp/torchaudio-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
# COPY DockerENV/torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl /tmp/torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl

# 创建 /home/lzz/aiflask 目录
RUN mkdir -p /home/lzz/aiflask
RUN mkdir -p /home/AIData/Datasets

# 复制本地的文件或文件夹到容器的指定路径
COPY . /home/lzz/aiflask/

# 安装 Anaconda
RUN bash /tmp/Anaconda3.sh -b -p /opt/anaconda3

# 配置环境变量，添加 Anaconda 到 PATH 中
ENV PATH=/opt/anaconda3/bin:$PATH
ENV PYTHONPATH=/home/lzz/aiflask

# 初始化 Conda 并创建环境
RUN /opt/anaconda3/bin/conda init bash

# 创建 Conda 环境并安装 Python 3.11
RUN conda create -n AIFlask python=3.11 -y

# 激活环境并将其设为默认环境
RUN echo "conda activate AIFlask" >> ~/.bashrc

#### 下载速度慢的话可以手动下载这几个包，然后手动安装（同上命令一起使用，但注意注释下面在线下载的命令）
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install /tmp/torch-2.4.1+cu124-cp311-cp311-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install /tmp/torchaudio-2.4.1+cu124-cp311-cp311-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install /tmp/torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用 conda run 安装 PyTorch（避免手动激活）

RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install flask_cors -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install ray -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install minio -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install ultralytics==8.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install gitpython -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN /opt/anaconda3/bin/conda run -n AIFlask pip3 install gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 删除安装包
RUN rm /tmp/Anaconda3.sh
RUN rm /tmp/torch-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
RUN rm /tmp/torchaudio-2.4.1+cu124-cp311-cp311-linux_x86_64.whl
RUN rm /tmp/torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl

RUN rm -rf /home/lzz/aiflask/DockerENV

# 设置默认启动命令
CMD ["bash"]
