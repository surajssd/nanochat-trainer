# nanochat-trainer

Infrastructure and packaging for running [Karpathy's nanochat](https://github.com/karpathy/nanochat) LLM training pipeline on Kubernetes with GPUs.

This repo does **not** contain ML training code. It provides a GPU-ready container image, a phased entrypoint script, Kubernetes manifests, and a pipeline runner that orchestrates the full training workflow end-to-end.

## Prerequisites

- A Kubernetes cluster with GPU nodes (tested on Azure `Standard_ND96isr_H100_v5` with 8x H100)
- `kubectl` configured and pointing at your cluster
- `docker` (with `buildx`) for building images
- `gh` CLI (GitHub CLI) for resolving the latest upstream nanochat commit

## Quick Start

### 1. Build and push the container image

```bash
make push
```

This clones the latest upstream nanochat, installs dependencies with `uv`, and pushes a tagged image to `quay.io/surajd/nanochat`.

### 2. Run the full pipeline

```bash
./run-pipeline.sh
```

That's it. The script creates the namespace and PVC, then runs all 8 phases sequentially, waiting for each to complete before starting the next.

### 3. Chat with your model

After the pipeline finishes, port-forward the chat UI to your machine:

```bash
kubectl port-forward -n nanochat deployment/nanochat-08-chat-serve 8000:8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Pipeline Phases

The training pipeline consists of 8 phases. Phases 1-7 run as Kubernetes Jobs; phase 8 runs as a long-lived Deployment.

| # | Phase | Description | GPUs | Timeout |
|---|-------|-------------|------|---------|
| 1 | `dataset` | Download training dataset (~17GB) | 0 | 1h |
| 2 | `tokenizer` | Train and evaluate BPE tokenizer | 0 | 30m |
| 3 | `base-train` | Pretrain base model (auto-resumes from checkpoints) | 8 | 4h |
| 4 | `base-eval` | Evaluate base model (CORE/BPB benchmarks) | 8 | 1h |
| 5 | `sft` | Supervised fine-tuning on chat data | 8 | 2h |
| 6 | `chat-eval` | Evaluate chat model (ChatCORE) | 8 | 1h |
| 7 | `report` | Generate final report | 0 | 10m |
| 8 | `chat-serve` | Serve interactive chat web UI | 1 | 5m |

All phases share a 256Gi PersistentVolumeClaim (`nanochat-data`) mounted at `/data/nanochat`, so checkpoints, datasets, and model artifacts persist across phases and restarts.

## Pipeline Runner

`run-pipeline.sh` orchestrates the phases on your Kubernetes cluster.

```bash
# Run the full pipeline
./run-pipeline.sh

# Resume from a specific phase (skips earlier phases)
./run-pipeline.sh -s base-train

# Run a single phase
./run-pipeline.sh -o sft

# Dry-run: see what would happen without applying anything
./run-pipeline.sh -n

# Tail pod logs live while waiting
./run-pipeline.sh -l -s sft
```

### Options

| Flag | Description |
|------|-------------|
| `-s, --start PHASE` | Start from PHASE, skipping earlier phases |
| `-o, --only PHASE` | Run a single phase and exit |
| `-n, --dry-run` | Print the plan without applying anything |
| `-l, --logs` | Tail pod logs while waiting for completion |
| `-h, --help` | Show help message |

## Building the Container Image

```bash
# Build locally
make build

# Build and push to registry
make push

# Pin to a specific upstream nanochat commit
make push NANOCHAT_VERSION=abc1234

# Cross-platform build
make push PLATFORM=linux/arm64
```

### Image Details

- **Base image:** `nvidia/cuda:13.1.1-devel-ubuntu24.04`
- **Python packaging:** [uv](https://github.com/astral-sh/uv) (`uv venv && uv sync --extra gpu`)
- **Tag format:** `<nanochat-sha>-<this-repo-sha>-<YYYYMMDDHHMMSS>`

## Configuration

All training parameters are configured via environment variables, set in the Kubernetes manifests.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPTH` | `24` | Number of transformer layers (the single "complexity dial") |
| `DEVICE_BATCH_SIZE` | `16` | Per-GPU batch size |
| `NUM_GPUS` | `8` | Number of GPUs for training phases |
| `CHAT_NUM_GPUS` | `1` | Number of GPUs for the chat-serve phase |
| `SAVE_EVERY` | `500` | Checkpoint save interval (steps) |
| `DATASET_SHARDS` | `170` | Number of dataset shards to download |
| `TARGET_PARAM_DATA_RATIO` | `9.5` | Parameter-to-data ratio for training |
| `WANDB_RUN` | `dummy` | Weights & Biases run name |
| `NANOCHAT_BASE_DIR` | `/data/nanochat` | Base directory for all data and checkpoints |

## Kubernetes Manifests

All manifests live in `k8s/single-node/`:

| File | Kind | Description |
|------|------|-------------|
| `pvc.yaml` | Namespace + PVC | Creates `nanochat` namespace and 256Gi PVC |
| `01-dataset.yaml` | Job | Downloads training data |
| `02-tokenizer.yaml` | Job | Trains BPE tokenizer |
| `03-base-train.yaml` | Job | Pretrains the base model |
| `04-base-eval.yaml` | Job | Evaluates the base model |
| `05-sft.yaml` | Job | Supervised fine-tuning |
| `06-chat-eval.yaml` | Job | Evaluates the chat model |
| `07-report.yaml` | Job | Generates final report |
| `08-chat-serve.yaml` | Deployment | Serves the chat web UI on port 8000 |

The chat-serve Deployment includes readiness and liveness probes on the `/health` endpoint to ensure the model is fully loaded before accepting traffic.

### Running Individual Phases Manually

You can apply any manifest directly:

```bash
# Apply namespace and PVC first
kubectl apply -f k8s/single-node/pvc.yaml

# Run a specific phase
kubectl apply -f k8s/single-node/05-sft.yaml

# Watch it
kubectl logs -n nanochat job/nanochat-05-sft -f

# Or deploy the chat UI
kubectl apply -f k8s/single-node/08-chat-serve.yaml
kubectl port-forward -n nanochat deployment/nanochat-08-chat-serve 8000:8000
```

## Troubleshooting

### Check phase status

```bash
# List all jobs
kubectl get jobs -n nanochat

# Check the chat-serve deployment
kubectl get deployment -n nanochat

# Describe a failing job
kubectl describe job -n nanochat nanochat-03-base-train

# View logs
kubectl logs -n nanochat job/nanochat-03-base-train
```

### Base training auto-resume

The `base-train` phase automatically detects existing checkpoints in `$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}/` and resumes from the latest one. If the job is interrupted, simply re-run it and training picks up where it left off.

### Storage

The PVC uses `managed-csi-premium` storage class with `ReadWriteOnce` access mode. For multi-node training, change the access mode to `ReadWriteMany` and use a storage class that supports it.

## File Structure

```
nanochat-trainer/
├── Dockerfile              # GPU container image build
├── Makefile                # Build/push automation
├── entrypoint.sh           # Phased training entrypoint
├── run-pipeline.sh         # K8s pipeline orchestrator
├── k8s/
│   └── single-node/
│       ├── pvc.yaml            # Namespace + PVC
│       ├── 01-dataset.yaml     # Dataset download
│       ├── 02-tokenizer.yaml   # Tokenizer training
│       ├── 03-base-train.yaml  # Base model pretraining
│       ├── 04-base-eval.yaml   # Base model evaluation
│       ├── 05-sft.yaml         # Supervised fine-tuning
│       ├── 06-chat-eval.yaml   # Chat model evaluation
│       ├── 07-report.yaml      # Report generation
│       └── 08-chat-serve.yaml  # Chat web UI deployment
├── CLAUDE.md               # Claude Code project instructions
├── LICENSE                 # MIT
└── README.md               # This file
```

## License

MIT
