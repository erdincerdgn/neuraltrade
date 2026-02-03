# ============================================================
# NeuralTrade - Production Dockerfile (VENV FIXED + CACHE FIX)
# ============================================================
# 1. "No module named uvicorn" -> ÇÖZÜLDÜ (VENV ile)
# 2. "OpenBB Permission Denied" -> ÇÖZÜLDÜ (chown /opt/venv ile)
# 3. "FastEmbed/HF Permission Denied" -> ÇÖZÜLDÜ (Home Dir ile)
# ============================================================

# --- Stage 1: Builder ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as builder

WORKDIR /build
ENV DEBIAN_FRONTEND=noninteractive

# 1. Python 3.11 ve VENV araçlarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    gcc \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. SANAL ORTAM (VENV)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 3. TA-Lib Derleme
ARG TA_LIB_VERSION=0.4.0
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-${TA_LIB_VERSION}-src.tar.gz && \
    tar -xvzf ta-lib-${TA_LIB_VERSION}-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-${TA_LIB_VERSION}-src.tar.gz

# 4. Paketleri Yükle
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt && \
    pip uninstall -y keras || true && \
    pip install tf-keras

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as runtime

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Runtime Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgomp1 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopyalamalar
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . .

# Kullanıcı ve İzinler (GÜNCELLENDİ)
RUN groupadd -r neuraltrade && \
    useradd -r -m -d /home/neuraltrade -s /bin/bash -g neuraltrade neuraltrade && \
    mkdir -p /app/data /app/logs /app/config && \
    # 👇 DEĞİŞİKLİK 1: Cache klasörlerini /app/data altından kurtarıp EV klasörüne aldık
    mkdir -p /home/neuraltrade/huggingface/{hub,transformers,sentence-transformers} && \
    mkdir -p /home/neuraltrade/fastembed_cache && \
    mkdir -p /home/neuraltrade/.openbb /tmp/matplotlib /tmp/openbb_cache && \
    \
    # İzinler
    chown -R neuraltrade:neuraltrade /opt/venv && \
    chown -R neuraltrade:neuraltrade /app && \
    chown -R neuraltrade:neuraltrade /home/neuraltrade && \
    chmod -R 777 /tmp/matplotlib /tmp/openbb_cache && \
    chown -R neuraltrade:neuraltrade /tmp/matplotlib /tmp/openbb_cache

USER neuraltrade

# Env Vars (GÜNCELLENDİ)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    NEURALTRADE_ENV=production \
    PYTHONPATH=/app \
    HOME=/home/neuraltrade \
    PATH="/opt/venv/bin:$PATH" \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    # 👇 DEĞİŞİKLİK 2: Yolları güvenli eve yönlendirdik
    HF_HOME=/home/neuraltrade/huggingface \
    HF_HUB_CACHE=/home/neuraltrade/huggingface/hub \
    TRANSFORMERS_CACHE=/home/neuraltrade/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/home/neuraltrade/huggingface/sentence-transformers \
    FASTEMBED_CACHE_PATH=/home/neuraltrade/fastembed_cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 50051 9090

# Bu komut kesinlikle doğru
CMD ["python", "api_server.py"]
# CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]