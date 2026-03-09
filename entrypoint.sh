#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# entrypoint.sh — phased training pipeline for nanochat
#
# Usage:  entrypoint.sh <phase> [extra-args...]
#
# Phases: dataset, tokenizer, base-train, base-eval, sft, chat-eval, report
#
# All configuration is via environment variables (see defaults below).
# Extra arguments after the phase name are forwarded to the upstream script.
# ---------------------------------------------------------------------------

# ---- Configurable defaults (override via K8s env) -------------------------
DEPTH="${DEPTH:-24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DATASET_SHARDS="${DATASET_SHARDS:-170}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-9.5}"
WANDB_RUN="${WANDB_RUN:-dummy}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/data/nanochat}"

IDENTITY_URL="https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "==> [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
}

# find_latest_checkpoint DIR
#   Scans DIR for model_*.pt files, prints the highest step number.
#   Returns 1 if no checkpoint is found.
find_latest_checkpoint() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return 1
    fi

    local latest=-1
    for f in "$dir"/model_*.pt; do
        [ -e "$f" ] || continue # guard against no matches
        local step
        step="$(basename "$f" .pt)" # model_01234
        step="${step#model_}"       # 01234
        step="$((10#$step))"        # strip leading zeros → 1234
        if [ "$step" -gt "$latest" ]; then
            latest="$step"
        fi
    done

    if [ "$latest" -eq -1 ]; then
        return 1
    fi
    echo "$latest"
}

# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

phase_dataset() {
    log "Downloading dataset (${DATASET_SHARDS} shards)..."
    python -m nanochat.dataset -n "${DATASET_SHARDS}" "$@"
}

phase_tokenizer() {
    log "Training tokenizer..."
    python -m scripts.tok_train "$@"
    log "Evaluating tokenizer..."
    python -m scripts.tok_eval
}

phase_base_train() {
    local checkpoint_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/d${DEPTH}"
    local resume_args=()

    local latest_step
    if latest_step="$(find_latest_checkpoint "$checkpoint_dir")"; then
        log "Found checkpoint at step ${latest_step} in ${checkpoint_dir}"
        resume_args+=("--resume-from-step=${latest_step}")
    else
        log "No checkpoint found, starting from scratch"
    fi

    log "Starting base training (depth=${DEPTH}, GPUs=${NUM_GPUS}, save_every=${SAVE_EVERY})..."
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        -m scripts.base_train -- \
        --depth="${DEPTH}" \
        --device-batch-size="${DEVICE_BATCH_SIZE}" \
        --target-param-data-ratio="${TARGET_PARAM_DATA_RATIO}" \
        --save-every="${SAVE_EVERY}" \
        --fp8 \
        --run="${WANDB_RUN}" \
        "${resume_args[@]}" \
        "$@"
}

phase_base_eval() {
    log "Evaluating base model..."
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        -m scripts.base_eval -- \
        --device-batch-size="${DEVICE_BATCH_SIZE}" \
        "$@"
}

phase_sft() {
    local identity_file="${NANOCHAT_BASE_DIR}/identity_conversations.jsonl"

    if [ ! -f "$identity_file" ]; then
        log "Downloading identity conversations..."
        curl -fSL -o "$identity_file" "$IDENTITY_URL"
    else
        log "Identity conversations already present, skipping download"
    fi

    log "Starting SFT training..."
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        -m scripts.chat_sft -- \
        --device-batch-size="${DEVICE_BATCH_SIZE}" \
        --run="${WANDB_RUN}" \
        "$@"
}

phase_chat_eval() {
    log "Evaluating chat model..."
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
        -m scripts.chat_eval -- \
        -i sft \
        "$@"
}

phase_report() {
    log "Generating report..."
    python -m nanochat.report generate "$@"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <phase> [extra-args...]"
    echo "Phases: dataset, tokenizer, base-train, base-eval, sft, chat-eval, report"
    exit 1
fi

PHASE="$1"
shift

case "$PHASE" in
dataset) phase_dataset "$@" ;;
tokenizer) phase_tokenizer "$@" ;;
base-train) phase_base_train "$@" ;;
base-eval) phase_base_eval "$@" ;;
sft) phase_sft "$@" ;;
chat-eval) phase_chat_eval "$@" ;;
report) phase_report "$@" ;;
*)
    echo "Error: unknown phase '${PHASE}'"
    echo "Phases: dataset, tokenizer, base-train, base-eval, sft, chat-eval, report"
    exit 1
    ;;
esac
