FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TORCH_EXTENSIONS_DIR=/app/torch_extensions

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git libgl1 libglib2.0-0 build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install PyTorch CUDA wheels + deps
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt && \
    pip3 install --no-cache-dir gunicorn

# App code
COPY . /app
RUN mkdir -p /app/models /app/data /app/torch_extensions

EXPOSE 8000
CMD ["bash", "-lc", "gunicorn -w 1 -k gthread -t 120 --bind 0.0.0.0:8000 app:app"]