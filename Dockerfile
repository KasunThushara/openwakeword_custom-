# ═══════════════════════════════════════════════════════════════════
#  Bumblebee Wake Word — Docker Image
#  Base:  ubuntu:22.04
#  Arch:  x86_64  |  Python: 3.10
# ═══════════════════════════════════════════════════════════════════

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System packages ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    libsndfile1 \
    libsndfile1-dev \
    libportaudio2 \
    portaudio19-dev \
    libasound2 \
    libasound2-dev \
    libasound2-plugins \
    alsa-utils \
    ffmpeg \
    build-essential \
    pkg-config \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════
#  Python dependencies — ORDER IS CRITICAL
#
#  The problem:  datasets / audiomentations pull in numpy >= 2.x
#  The fix:      pin numpy < 2.0 as the VERY LAST pip install step,
#                with --force-reinstall so it overrides whatever
#                any earlier package installed.
# ═══════════════════════════════════════════════════════════════════

# 1 ── PyTorch CPU (must come before openwakeword)
RUN pip install \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 2 ── openWakeWord
RUN pip install openwakeword==0.6.0

# 3 ── All remaining packages
#      (some of these will upgrade numpy to 2.x — that is OK here,
#       we fix it in the next step)
RUN pip install \
    piper-tts==1.4.2 \
    soundfile==0.14.0 \
    librosa==0.11.0 \
    audiomentations==0.43.1 \
    datasets==5.0.0 \
    tqdm \
    scikit-learn \
    pyyaml \
    requests \
    onnx \
    onnxruntime \
    sounddevice \
    flask \
    websockets

# 4 ── Pin numpy < 2.0  ← MUST be last, AFTER all other packages
#      --force-reinstall overwrites the numpy 2.x that datasets/audiomentations pulled in
RUN pip install "numpy<2.0" --force-reinstall

# 5 ── Re-pin torch AFTER numpy fix to ensure nothing downgraded it
RUN pip install \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall

# ── ALSA null config (suppresses "no soundcards" when mic not mounted) ──
RUN echo 'pcm.!default { type null }'  > /etc/asound.conf \
 && echo 'ctl.!default { type null }' >> /etc/asound.conf

# ── Copy project source ───────────────────────────────────────────
COPY config.py                  ./
COPY step1_generate_clips.py    ./
COPY step2_get_negatives.py     ./
COPY step3_extract_features.py  ./
COPY step4_train.py             ./
COPY step5_test.py              ./
COPY diagnose.py                ./
COPY app.py                     ./
COPY bumblebee_server.py        ./
COPY bumblebee_face.html        ./

RUN mkdir -p data/positive data/negative features model voices

EXPOSE 5000
EXPOSE 8767

# ── Verify: numpy must be 1.26.x and torch bridge must work ───────
RUN python3 -c "\
import numpy, torch, openwakeword; \
print('numpy :', numpy.__version__); \
assert numpy.__version__.startswith('1.'), 'numpy is NOT < 2.0 — build failed'; \
print('torch :', torch.__version__); \
arr = torch.from_numpy(numpy.zeros(10)); \
print('bridge: OK')"

CMD ["python3", "app.py"]