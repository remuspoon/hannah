FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.12 via deadsnakes
RUN apt-get update && apt-get install -y \
    software-properties-common curl git wget openssh-server \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# PyTorch first (needs special index URL for cu126 build)
RUN pip install --no-cache-dir \
    torch==2.12.0+cu126 \
    torchvision==0.27.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Remaining packages
COPY requirements-llm.txt /tmp/requirements-llm.txt
RUN pip install --no-cache-dir -r /tmp/requirements-llm.txt

# SSH setup for RunPod
RUN mkdir /var/run/sshd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

# Jupyter config -- no password/token, listen on all interfaces, allow RunPod proxy
RUN jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_origin_pat = '.*'" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.tornado_settings = {'headers': {'Content-Security-Policy': \"frame-ancestors * 'self'\"}}" >> /root/.jupyter/jupyter_lab_config.py

WORKDIR /workspace

EXPOSE 8888 22

# Start SSH + Jupyter on container launch
CMD service ssh start && jupyter lab --port=8888 --no-browser --allow-root
