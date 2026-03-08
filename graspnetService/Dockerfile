# Contact-GraspNet FastAPI inference service
# Tested stack per NVlabs README: Python 3.7, TensorFlow 2.2, CUDA 10.1
# Build: docker build -t graspnet-service .
# Run:   docker compose up -d  (see docker-compose.yml)

# TF 2.2.0 requires CUDA 10.1 + cuDNN 7 (official tested stack per NVlabs README)
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYOPENGL_PLATFORM=egl

# System deps — Ubuntu 18.04 ships Python 3.7 in the default repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 python3.7-dev python3-pip python3.7-venv \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.7 specifically
RUN curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# Make python3 and python point to python3.7
# Ubuntu 18.04 ships python3.6 which may own /usr/bin/python3 — force-override it
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.7 /usr/bin/python

WORKDIR /app

# Clone NVlabs Contact-GraspNet
RUN git clone https://github.com/NVlabs/contact_graspnet.git .

# Install Python deps matching NVlabs tested versions
RUN python3.7 -m pip install --upgrade pip setuptools wheel && \
    python3.7 -m pip install --no-cache-dir \
        tensorflow==2.2.0 \
        numpy==1.19.5 \
        scipy \
        scikit-learn \
        trimesh \
        Pillow \
        matplotlib \
        tqdm \
        "opencv-python-headless==4.8.1.78" \
        "fastapi==0.103.2" \
        "uvicorn==0.22.0" \
        "pydantic==1.10.13" \
        "protobuf==3.20.3" \
        "pyyaml==5.4.1"

# Compile PointNet2 TF ops — required for inference
RUN bash compile_pointnet_tfops.sh

# Copy our FastAPI server wrapper
COPY graspnet_server.py /app/graspnet_server.py

# Checkpoints are mounted at runtime via docker-compose.yml volume
VOLUME ["/app/checkpoints"]

EXPOSE 8766

CMD ["python3.7", "graspnet_server.py"]
