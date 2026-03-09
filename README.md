# nanochat-trainer

Infrastructure and packaging for running [Karpathy's nanochat](https://github.com/karpathy/nanochat) LLM training pipeline on Kubernetes with GPUs.

This repo does **not** contain ML training code. It provides a GPU-ready container image, a phased entrypoint script, Kubernetes manifests, and an [Argo Workflow](https://argoproj.github.io/workflows/) that orchestrates the full training pipeline end-to-end.

## Prerequisites

- A Kubernetes cluster with GPU nodes (tested on Azure `Standard_ND96isr_H100_v5` with 8x H100)
- [Argo Workflows](https://argoproj.github.io/workflows/quick-start/) installed on the cluster
- `kubectl` configured and pointing at your cluster
- `docker` (with `buildx`) for building images
- `gh` CLI (GitHub CLI) for resolving the latest upstream nanochat commit

## Quick Start

### 1. Build and push the container image

```bash
make push
```

This clones the latest upstream nanochat, installs dependencies with `uv`, and pushes a tagged image to `quay.io/surajd/nanochat`.

### 2. Create the namespace and PVC

```bash
kubectl apply -f k8s/single-node/pvc.yaml
```

### 3. Run the training pipeline

```bash
kubectl create -f k8s/single-node/workflow.yaml
```

This submits the Argo Workflow which runs all 7 training phases sequentially (dataset → tokenizer → base-train → base-eval → sft → chat-eval → report), waiting for each to complete before starting the next.

Monitor progress:

```bash
kubectl get workflow -n nanochat
kubectl describe workflow -n nanochat -l workflows.argoproj.io/workflow
```

### 4. Chat with your model

After the workflow completes, deploy the chat UI and port-forward:

```bash
kubectl apply -f k8s/single-node/chat-serve.yaml
kubectl port-forward -n nanochat deployment/nanochat-08-chat-serve 8000:8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Pipeline Phases

The training pipeline consists of 7 phases defined as sequential steps in an Argo Workflow. Each phase creates a Kubernetes Job with `generateName` for clean re-runs.

| # | Phase | Description | GPUs | Timeout |
|---|-------|-------------|------|---------|
| 1 | `dataset` | Download training dataset (~17GB) | 0 | 1h |
| 2 | `tokenizer` | Train and evaluate BPE tokenizer | 0 | 30m |
| 3 | `base-train` | Pretrain base model (auto-resumes from checkpoints) | 8 | 4h |
| 4 | `base-eval` | Evaluate base model (CORE/BPB benchmarks) | 8 | 1h |
| 5 | `sft` | Supervised fine-tuning on chat data | 8 | 2h |
| 6 | `chat-eval` | Evaluate chat model (ChatCORE) | 8 | 1h |
| 7 | `report` | Generate final report | 0 | 10m |

The chat-serve Deployment (phase 8) is a standalone manifest applied separately after the workflow completes — it's a long-lived interactive application, not a training phase.

All phases share a 256Gi PersistentVolumeClaim (`nanochat-data1`) mounted at `/data/nanochat`, so checkpoints, datasets, and model artifacts persist across phases and restarts.

## Argo Workflow Usage

### Submit and monitor

```bash
# Submit the workflow
kubectl create -f k8s/single-node/workflow.yaml

# List workflows
kubectl get workflows -n nanochat

# Watch workflow progress
kubectl get workflows -n nanochat -w

# Get detailed workflow status
kubectl describe workflow -n nanochat <workflow-name>

# View logs for a specific phase's job
kubectl get jobs -n nanochat
kubectl logs -n nanochat job/<job-name>
```

### Retry and re-run

```bash
# Re-submit a fresh run (generateName creates a new workflow each time)
kubectl create -f k8s/single-node/workflow.yaml

# Delete a workflow and its child resources (owner references clean up jobs automatically)
kubectl delete workflow -n nanochat <workflow-name>

# Delete all workflows
kubectl delete workflows -n nanochat --all
```

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

All training parameters are configured via environment variables, set in the workflow manifest.

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
| `workflow.yaml` | Argo Workflow | Full training pipeline (7 phases as sequential steps) |
| `chat-serve.yaml` | Deployment | Serves the chat web UI on port 8000 (deployed separately) |

The chat-serve Deployment includes readiness and liveness probes on the `/health` endpoint to ensure the model is fully loaded before accepting traffic.

## Troubleshooting

### Check workflow status

```bash
# List all workflows
kubectl get workflows -n nanochat

# Get detailed workflow status
kubectl describe workflow -n nanochat <workflow-name>

# List child jobs created by the workflow
kubectl get jobs -n nanochat

# View logs for a specific phase's job
kubectl logs -n nanochat job/<job-name>
```

### Retry a failed phase

```bash
# Delete the failed workflow (owner references clean up child jobs)
kubectl delete workflow -n nanochat <workflow-name>

# Re-submit a fresh run
kubectl create -f k8s/single-node/workflow.yaml
```

### Check the chat-serve deployment

```bash
kubectl get deployment -n nanochat
kubectl describe deployment -n nanochat nanochat-08-chat-serve
kubectl logs -n nanochat deployment/nanochat-08-chat-serve
```

### Base training auto-resume

The `base-train` phase automatically detects existing checkpoints in `$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}/` and resumes from the latest one. If the job is interrupted, simply re-run the workflow and training picks up where it left off.

### Storage

The PVC uses `managed-csi-premium` storage class with `ReadWriteOnce` access mode. For multi-node training, change the access mode to `ReadWriteMany` and use a storage class that supports it.

## File Structure

```
nanochat-trainer/
├── Dockerfile              # GPU container image build
├── Makefile                # Build/push automation
├── entrypoint.sh           # Phased training entrypoint
├── k8s/
│   └── single-node/
│       ├── pvc.yaml            # Namespace + PVC
│       ├── workflow.yaml       # Argo Workflow (full training pipeline)
│       └── chat-serve.yaml  # Chat web UI deployment
├── CLAUDE.md               # Claude Code project instructions
├── LICENSE                 # MIT
└── README.md               # This file
```

## License

MIT
