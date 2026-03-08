FROM nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Pin to a specific commit/tag/branch for reproducibility
ARG NANOCHAT_VERSION=master
RUN git clone https://github.com/karpathy/nanochat.git . && \
    git checkout ${NANOCHAT_VERSION} && \
    rm -rf .git

RUN uv venv && uv sync --extra gpu

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV OMP_NUM_THREADS=1
ENV NANOCHAT_BASE_DIR=/data/nanochat
ENV WANDB_RUN=dummy
