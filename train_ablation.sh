#!/bin/bash
# Training ablation for peg insertion: 3 architectures × 2 prediction horizons.
# Run from inside the diffusion_policy repo on a GPU node.
# Usage: bash train_ablation.sh
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO_ROOT="$(pwd)"
LOG_DIR="${REPO_ROOT}/logs/train_ablation"
mkdir -p "${LOG_DIR}"

# ── Wandb ──
if ! python -c "import wandb; wandb.api.api_key" 2>/dev/null; then
    echo "wandb not logged in. Run: wandb login"
    exit 1
fi
export WANDB_PROJECT="peg_insertion_ablation"

# ── Detect available GPUs ──
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

# ── Grid: 3 architectures × 2 prediction horizons = 6 runs ──
configs=(
    "train_mlp_sim2real_state_workspace"
    "train_mlp_sim2real_state_workspace"
    "train_transformer_sim2real_state_workspace"
    "train_transformer_sim2real_state_workspace"
    "train_diffusion_sim2real_state_workspace"
    "train_diffusion_sim2real_state_workspace"
)
action_steps=(8 16 8 16 8 16)
horizons=(12 20 12 20 12 20)

NUM_JOBS=${#configs[@]}

run_job() {
    local idx=$1
    local gpu=$2
    local config=${configs[$idx]}
    local n_act=${action_steps[$idx]}
    local hor=${horizons[$idx]}
    local exp="h${n_act}"
    local out="${REPO_ROOT}/outputs/train_ablation/${config}/${exp}"
    local log="${LOG_DIR}/${config}_${exp}.log"
    mkdir -p "${out}"

    echo "[Job ${idx}] GPU=${gpu}  ${config}  n_action_steps=${n_act}  horizon=${hor}"

    CUDA_VISIBLE_DEVICES="${gpu}" python "${REPO_ROOT}/train.py" \
        --config-name="${config}" \
        n_action_steps="${n_act}" \
        horizon="${hor}" \
        training.seed=42 \
        training.device="cuda:0" \
        training.num_epochs=1250 \
        logging.project="${WANDB_PROJECT}" \
        logging.group="${config}" \
        exp_name="${exp}" \
        output_dir="${out}" \
        > "${log}" 2>&1
}

# ── Launch jobs round-robin across GPUs ──
pids=()

for ((i=0; i<NUM_JOBS; i++)); do
    gpu=$((i % NUM_GPUS))

    if [ ${#pids[@]} -ge ${NUM_GPUS} ]; then
        wait_idx=$((i - NUM_GPUS))
        echo "Waiting for Job ${wait_idx} (PID ${pids[$wait_idx]})..."
        wait "${pids[$wait_idx]}"
        exit_code=$?
        if [ ${exit_code} -ne 0 ]; then
            echo "WARNING: Job ${wait_idx} exited with code ${exit_code}. See ${LOG_DIR}/"
        else
            echo "Job ${wait_idx} complete."
        fi
    fi

    run_job "${i}" "${gpu}" &
    pids+=($!)
done

echo "Waiting for remaining jobs..."
for ((i=0; i<NUM_JOBS; i++)); do
    wait "${pids[$i]}" 2>/dev/null || true
done

echo ""
echo "All ${NUM_JOBS} training runs complete."
echo "Logs:        ${LOG_DIR}/"
echo "Checkpoints: ${REPO_ROOT}/outputs/train_ablation/"
echo "Wandb:       https://wandb.ai -- project: ${WANDB_PROJECT}"
