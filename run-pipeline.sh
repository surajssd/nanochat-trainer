#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run-pipeline.sh — apply nanochat training phases sequentially on K8s
# ---------------------------------------------------------------------------

SCRIPT_NAME="$(basename "$0")"
MANIFEST_DIR="$(cd "$(dirname "$0")/k8s/single-node" && pwd)"
NAMESPACE="nanochat"

# Ordered list of phases: name|manifest|resource-name|timeout|description|kind
PHASES=(
    "dataset|01-dataset.yaml|nanochat-01-dataset|1h|📦 Download training dataset (~17GB)|job"
    "tokenizer|02-tokenizer.yaml|nanochat-02-tokenizer|30m|🔤 Train and evaluate BPE tokenizer|job"
    "base-train|03-base-train.yaml|nanochat-03-base-train|4h|🧠 Pretrain base model (auto-resumes)|job"
    "base-eval|04-base-eval.yaml|nanochat-04-base-eval|1h|📊 Evaluate base model (CORE/BPB)|job"
    "sft|05-sft.yaml|nanochat-05-sft|2h|💬 Supervised fine-tuning|job"
    "chat-eval|06-chat-eval.yaml|nanochat-06-chat-eval|1h|🏆 Evaluate chat model (ChatCORE)|job"
    "report|07-report.yaml|nanochat-07-report|10m|📝 Generate final report|job"
    "chat-serve|08-chat-serve.yaml|nanochat-08-chat-serve|5m|🌐 Serve chat web UI|deployment"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ANSI colours (disabled if stdout is not a terminal)
if [ -t 1 ]; then
    BOLD=$'\033[1m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    CYAN=$'\033[36m'
    RESET=$'\033[0m'
else
    BOLD="" GREEN="" YELLOW="" RED="" CYAN="" RESET=""
fi

log() { echo "🚀 ${BOLD}==>${RESET} [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
info() { echo "    ${CYAN}$*${RESET}"; }
ok() { echo "    ${GREEN}✅ $*${RESET}"; }
warn() { echo "    ${YELLOW}⚠️  $*${RESET}"; }
err() { echo "    ${RED}❌ $*${RESET}" >&2; }

# Extract the phase name from a PHASES entry
phase_name() { echo "$1" | cut -d'|' -f1; }

usage() {
    cat <<EOF
${BOLD}Usage:${RESET}  ${SCRIPT_NAME} [options] [start-phase]

Apply nanochat training phases as sequential K8s Jobs, waiting for each to
complete before starting the next. Data is persisted on a shared PVC so phases
(and restarts) pick up where they left off.

${BOLD}Phases:${RESET}
EOF
    for entry in "${PHASES[@]}"; do
        IFS='|' read -r name _ _ timeout desc _ <<<"$entry"
        printf "  ${CYAN}%-12s${RESET} %s ${YELLOW}(timeout: %s)${RESET}\n" "$name" "$desc" "$timeout"
    done
    cat <<EOF

${BOLD}Options:${RESET}
  -s, --start PHASE   Start from PHASE, skipping earlier phases
  -o, --only  PHASE   Run a single phase and exit
  -n, --dry-run       Print what would be applied without running anything
  -l, --logs          Tail pod logs while waiting (instead of kubectl wait)
  -h, --help          Show this help message

${BOLD}Examples:${RESET}
  ${SCRIPT_NAME}                    # run full pipeline
  ${SCRIPT_NAME} -s base-train      # resume from base-train
  ${SCRIPT_NAME} -o sft             # run only the SFT phase
  ${SCRIPT_NAME} -n                 # dry-run: show what would happen
  ${SCRIPT_NAME} -l -s sft          # run from SFT, tailing logs live
EOF
}

# Resolve a phase name to its index (0-based). Exits on invalid name.
resolve_phase_index() {
    local target="$1"
    for i in "${!PHASES[@]}"; do
        if [ "$(phase_name "${PHASES[$i]}")" = "$target" ]; then
            echo "$i"
            return
        fi
    done
    err "Unknown phase '${target}'"
    echo ""
    echo "Valid phases:"
    for entry in "${PHASES[@]}"; do
        echo "  $(phase_name "$entry")"
    done
    exit 1
}

# Wait for a Job to complete, optionally tailing logs.
wait_for_job() {
    local job_name="$1" timeout="$2"

    if [ "$TAIL_LOGS" = true ]; then
        # Stream logs in the background; kill on exit
        (
            # Give the pod a moment to start
            sleep 5
            kubectl logs -n "${NAMESPACE}" "job/${job_name}" -f --all-containers 2>/dev/null || true
        ) &
        local log_pid=$!
        trap 'kill $log_pid 2>/dev/null || true' RETURN

        kubectl wait --for=condition=complete "job/${job_name}" \
            -n "${NAMESPACE}" --timeout="${timeout}"

        kill "$log_pid" 2>/dev/null || true
        trap - RETURN
    else
        kubectl wait --for=condition=complete "job/${job_name}" \
            -n "${NAMESPACE}" --timeout="${timeout}"
    fi
}

# Wait for a Deployment to become Available.
wait_for_deployment() {
    local deploy_name="$1" timeout="$2"

    if [ "$TAIL_LOGS" = true ]; then
        (
            sleep 5
            kubectl logs -n "${NAMESPACE}" "deployment/${deploy_name}" -f --all-containers 2>/dev/null || true
        ) &
        local log_pid=$!
        trap 'kill $log_pid 2>/dev/null || true' RETURN

        kubectl wait --for=condition=Available "deployment/${deploy_name}" \
            -n "${NAMESPACE}" --timeout="${timeout}"

        kill "$log_pid" 2>/dev/null || true
        trap - RETURN
    else
        kubectl wait --for=condition=Available "deployment/${deploy_name}" \
            -n "${NAMESPACE}" --timeout="${timeout}"
    fi
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
START_INDEX=0
END_INDEX=$((${#PHASES[@]} - 1))
DRY_RUN=false
TAIL_LOGS=false

while [ $# -gt 0 ]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    -n | --dry-run)
        DRY_RUN=true
        shift
        ;;
    -l | --logs)
        TAIL_LOGS=true
        shift
        ;;
    -s | --start)
        [ $# -ge 2 ] || {
            err "--start requires a phase name"
            exit 1
        }
        START_INDEX="$(resolve_phase_index "$2")"
        shift 2
        ;;
    -o | --only)
        [ $# -ge 2 ] || {
            err "--only requires a phase name"
            exit 1
        }
        idx="$(resolve_phase_index "$2")"
        START_INDEX="$idx"
        END_INDEX="$idx"
        shift 2
        ;;
    -*)
        err "Unknown option: $1"
        echo ""
        usage
        exit 1
        ;;
    *)
        # Positional arg: treat as start phase (backward compat)
        START_INDEX="$(resolve_phase_index "$1")"
        shift
        ;;
    esac
done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Print plan
echo ""
log "📋 Pipeline plan:"
for i in "${!PHASES[@]}"; do
    IFS='|' read -r name _ job_name timeout desc _ <<<"${PHASES[$i]}"
    if [ "$i" -lt "$START_INDEX" ] || [ "$i" -gt "$END_INDEX" ]; then
        info "  ⏭️  skip   ${name}"
    else
        info "  ▶️  run    ${name}  →  ${desc}  (timeout: ${timeout})"
    fi
done
echo ""

if [ "$DRY_RUN" = true ]; then
    log "🔍 Dry-run mode — no changes applied"
    exit 0
fi

# Ensure namespace and PVC exist
log "💾 Applying namespace and PVC..."
kubectl apply -f "${MANIFEST_DIR}/pvc.yaml"
echo ""

for i in "${!PHASES[@]}"; do
    if [ "$i" -lt "$START_INDEX" ] || [ "$i" -gt "$END_INDEX" ]; then
        continue
    fi

    IFS='|' read -r name manifest job_name timeout desc kind <<<"${PHASES[$i]}"
    kind="${kind:-job}"

    log "Phase: ${name} — ${desc}"
    info "Manifest: ${manifest}"
    info "Timeout:  ${timeout}"

    kubectl apply -f "${MANIFEST_DIR}/${manifest}"

    if [ "$kind" = "deployment" ]; then
        if ! wait_for_deployment "$job_name" "$timeout"; then
            err "${job_name} deployment did not become available within ${timeout}"
            info "Check logs:        kubectl logs -n ${NAMESPACE} deployment/${job_name}"
            info "Describe deployment: kubectl describe deployment -n ${NAMESPACE} ${job_name}"
            exit 1
        fi

        ok "${name} is running"
        echo ""
        info "${BOLD}🌐 Chat web UI is ready!${RESET}"
        info "Run the following to access it from your machine:"
        echo ""
        info "  kubectl port-forward -n ${NAMESPACE} deployment/${job_name} 8000:8000"
        echo ""
        info "Then open ${CYAN}http://localhost:8000${RESET} in your browser."
    else
        if ! wait_for_job "$job_name" "$timeout"; then
            err "${job_name} did not complete within ${timeout}"
            info "Check logs:   kubectl logs -n ${NAMESPACE} job/${job_name}"
            info "Describe job: kubectl describe job -n ${NAMESPACE} ${job_name}"
            exit 1
        fi

        ok "${name} completed"
    fi
    echo ""
done

log "${GREEN}🎉 All phases completed successfully!${RESET}"
