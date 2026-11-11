# syntax=docker/dockerfile:1.4
# ==============================================================================
# Project CryptaVision — Multi-Stage CUDA & GStreamer Dockerfile
# ==============================================================================
# Architecture:
#   Stage 1 (Builder): Compiles custom C++ GStreamer base transform filter
#                      (crypta_frame_filter.so) with GCC and CMake.
#   Stage 2 (Runtime): Assembles NVIDIA CUDA 12.1 runtime, GStreamer plugins,
#                      Python 3.12 virtual environment, PyTorch, TenSEAL,
#                      FastAPI, and application source code.
# ==============================================================================

# ── STAGE 1: C++ Plugin Builder ───────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install C++ toolchain, CMake, and GStreamer development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/plugins
COPY plugins/ .

# Compile optimized C++ GStreamer base transform plugin
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)


# ── STAGE 2: Production CUDA & Python Runtime ─────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    GST_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/gstreamer-1.0" \
    MODEL_CACHE_DIR="/models" \
    TORCH_HOME="/models/torch" \
    HUGGINGFACE_HUB_CACHE="/models/huggingface" \
    YOLO_CONFIG_DIR="/models/ultralytics" \
    PORT=8000

# Install software-properties, curl, ca-certificates, and deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    # GStreamer 1.0 Runtime Stack
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    # GUI and media libraries for OpenCV
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install compiled C++ GStreamer filter from Stage 1 Builder
COPY --from=builder /build/plugins/build/libcrypta_frame_filter.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libcrypta_frame_filter.so

# Create isolated Python 3.12 virtual environment
RUN python3.12 -m venv /opt/venv

# ── Layer Caching: Install pip dependencies before copying application code ───
WORKDIR /app
COPY requirements-ai.txt requirements-crypto.txt requirements-api.txt /app/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements-api.txt \
        -r requirements-crypto.txt \
        -r requirements-ai.txt

# Create cache directory for models and prepare permissions
RUN mkdir -p /models/torch /models/huggingface /models/ultralytics && \
    chmod -R 777 /models

# Copy application source code
COPY pipeline/ /app/pipeline/
COPY config/ /app/config/
COPY crypto_engine.py vault_manager.py main.py /app/

# Create non-root application user for container hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# Docker healthcheck pinging FastAPI /health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
