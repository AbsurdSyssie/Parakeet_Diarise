FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/app/.cache/pip \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    ffmpeg \
    libsndfile1 \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
 && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m ensurepip \
 && python3.11 -m pip install -U pip

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3.11 -m pip install --upgrade \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    -r /app/requirements.txt

COPY . /app

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
