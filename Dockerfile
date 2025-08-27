# --------------------------------------------------------------
# Basis‑Image (wie von dir angegeben)
# --------------------------------------------------------------
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# --------------------------------------------------------------
# 0️⃣  USER‑Umgebung – das ist das Kernstück
# --------------------------------------------------------------
WORKDIR /workspace
ENV HOME=/workspace                         # <‑‑ wichtig
ENV PYTHONUSERBASE=${HOME}/.local
ENV PATH=${PYTHONUSERBASE}/bin:${PATH}
ENV PYTHONPATH=${HOME}:${PYTHONPATH}
ENV TRITON_CACHE_DIR=${HOME}/triton_cache
ENV TMPDIR=${HOME}/temp
ENV CCACHE_DIR=${HOME}/.ccache
ENV CCACHE_MAXSIZE=15G

# --------------------------------------------------------------
# 1️⃣  System‑Pakete (apt) + Python 3.12
# --------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common wget git build-essential cmake ninja-build \
        llvm-dev zlib1g-dev libzstd-dev libedit-dev libxml2-dev ccache \
        git-lfs \                               # <-- NEW: LFS für das Modell
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# LFS initialisieren (nur ein Mal nötig)
RUN git lfs install

# --------------------------------------------------------------
# 2️⃣  Standard‑Python‑Links auf 3.12 setzen
# --------------------------------------------------------------
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# --------------------------------------------------------------
# 3️⃣  pip für Python 3.12 (installiert in $PYTHONUSERBASE)
# --------------------------------------------------------------
RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --root-user-action=ignore --user && \
    rm -f get-pip.py && \
    pip3 --version

# --------------------------------------------------------------
# 4️⃣  Grund‑Python‑Pakete – **mit --user** in Workspace
# --------------------------------------------------------------
RUN pip3 install --user --no-cache-dir \
        numpy pytest scipy matplotlib pandas huggingface_hub

# --------------------------------------------------------------
# 5️⃣  Alte Triton‑Version entfernen (falls vorhanden)
# --------------------------------------------------------------
RUN pip3 uninstall -y triton || true

# --------------------------------------------------------------
# 6️⃣  Repos klonen  + 20 B‑Modell holen
# --------------------------------------------------------------
RUN git clone https://github.com/openai/gpt-oss.git ${HOME}/gpt-oss \
    && git clone https://github.com/triton-lang/triton.git ${HOME}/triton \
    \
    # ----- 20 B‑Modell von Hugging‑Face (Git‑LFS) -----
    && git clone https://huggingface.co/openai/gpt-oss-20b ${HOME}/gpt-oss-20b \
    && cd ${HOME}/gpt-oss-20b \
    && git lfs pull \
    && cd ${HOME}

# --------------------------------------------------------------
# 7️⃣  Triton bauen & installieren (user‑base)
# --------------------------------------------------------------
WORKDIR ${HOME}/triton
ENV MAX_JOBS=$(nproc)                # alle CPU‑Kerne nutzen
RUN pip3 install --user -r python/requirements.txt && \
    pip3 install --user -e . --no-build-isolation --verbose && \
    pip3 install --user -e python/triton_kernels

# --------------------------------------------------------------
# 8️⃣  GPT‑OSS mit Triton‑Support installieren
# --------------------------------------------------------------
WORKDIR ${HOME}/gpt-oss
RUN pip3 install --user -e ".[triton]"

# --------------------------------------------------------------
# 9️⃣  CUDA / Torch Optimierungen
# --------------------------------------------------------------
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# (weitere Torch‑/Triton‑Variablen können hier ergänzt werden)

# --------------------------------------------------------------
# 🔟  Startup‑Script (Info‑Print + Beispiel‑Aufruf)
# --------------------------------------------------------------
WORKDIR ${HOME}
RUN echo '#!/usr/bin/env bash' > startup.sh && \
    echo 'echo "=== GPT‑OSS + Triton + 20B model ready ==="' >> startup.sh && \
    echo 'echo "Python version: $(python --version)"' >> startup.sh && \
    echo 'echo "Triton cache dir : $TRITON_CACHE_DIR"' >> startup.sh && \
    echo 'echo "CCACHE dir       : $CCACHE_DIR"' >> startup.sh && \
    echo '' >> startup.sh && \
    echo 'echo "Model location:"' >> startup.sh && \
    echo 'echo "  ${HOME}/gpt-oss-20b"' >> startup.sh && \
    echo '' >> startup.sh && \
    echo 'echo "Run a quick generation example (triton backend):"' >> startup.sh && \
    echo 'cd ${HOME}/gpt-oss && \\' >> startup.sh && \
    echo 'python -m gpt_oss.generate \\' >> startup.sh && \
    echo '    --backend triton \\' >> startup.sh && \
    echo '    --model openai/gpt-oss-20b \\' >> startup.sh && \
    echo '    --prompt "Explain quantum mechanics in two sentences."' >> startup.sh && \
    echo '' >> startup.sh && \
    echo 'exec "$@"' >> startup.sh && \
    chmod +x startup.sh

# --------------------------------------------------------------
# 👉  Hinweis für das Runtime‑Starten (Docker‑Run‑Optionen)
# --------------------------------------------------------------
# Wenn du den Container startest, kann es je nach Host‑Hardware nötig sein,
# den Shared‑Memory zu vergrößern, z. B.:  
#   docker run --shm-size=1g --ulimit memlock=-1:-1  <image> …
# --------------------------------------------------------------

CMD ["/workspace/startup.sh", "bash"]
