#!/usr/bin/env bash
# Multi-turn supervised fine-tuning for SWE-style trajectories on top of verl's FSDP SFT trainer.
# Uses the same tokenizer and vocabulary as the base checkpoint (model.partial_pretrain).
# Full model fine-tune: keep LORA_RANK=0 (trains all parameters; model size unchanged).
#
# Mitigate forgetting other capabilities by:
#   - mixing general instruct data (see SWE_MIX_PARQUET and HYDRA_EXTRA),
#   - using a moderate learning rate and early stopping via val loss / small epoch count.
#
# After SFT, you can run RL with GPPO using the same codebase (recipe/dapo/*.sh), pointing
# actor_rollout_ref.model.path at the saved SFT checkpoint.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- Required local paths (export or edit here) ----
SWE_MODEL_PATH="${SWE_MODEL_PATH:-/path/to/your/Qwen-32B-Instruct}"
SWE_TRAIN_PARQUET="${SWE_TRAIN_PARQUET:-/path/to/swe_train.parquet}"
SWE_VAL_PARQUET="${SWE_VAL_PARQUET:-/path/to/swe_val.parquet}"

# Optional: second parquet merged into training (e.g. general chat / instruction following).
# Leave empty to train only on SWE_TRAIN_PARQUET.
SWE_MIX_PARQUET="${SWE_MIX_PARQUET:-}"

# ---- Cluster ----
NUM_GPUS="${NUM_GPUS:-8}"
# Sequence parallel (Ulysses). Use >1 for very long max_length on large models.
SP_SIZE="${SP_SIZE:-1}"

# ---- Data format ----
# Parquet column must be a list of {"role": "...", "content": "..."} (Qwen/chat templates).
MESSAGES_KEY="${MESSAGES_KEY:-messages}"

# ---- Training hyperparameters (tune for 32B + your GPU memory) ----
# Global batch before internal DP split; safe default ~1 sample per GPU when SP_SIZE=1.
GLOBAL_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE:-${NUM_GPUS}}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION="${TRUNCATION:-left}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_DIR="${SAVE_DIR:-${HOME}/ckpts/swe-qwen32b-multiturn-sft}"

# Qwen often needs trust_remote_code
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
# Strongly recommended for 32B full FT
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
LORA_RANK="${LORA_RANK:-0}"

# Extra Hydra overrides (space-separated), e.g.:
#   HYDRA_EXTRA="trainer.logger=['console','wandb'] optim.weight_decay=0"
HYDRA_EXTRA="${HYDRA_EXTRA:-}"


if [[ ! -f "${SWE_MODEL_PATH}/config.json" ]]; then
  echo "ERROR: SWE_MODEL_PATH must be a Hugging Face model directory containing config.json: ${SWE_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SWE_TRAIN_PARQUET}" ]]; then
  echo "ERROR: SWE_TRAIN_PARQUET not found: ${SWE_TRAIN_PARQUET}" >&2
  exit 1
fi
if [[ ! -f "${SWE_VAL_PARQUET}" ]]; then
  echo "ERROR: SWE_VAL_PARQUET not found: ${SWE_VAL_PARQUET}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

if [[ -n "${SWE_MIX_PARQUET}" ]]; then
  if [[ ! -f "${SWE_MIX_PARQUET}" ]]; then
    echo "ERROR: SWE_MIX_PARQUET set but file missing: ${SWE_MIX_PARQUET}" >&2
    exit 1
  fi
  # Hydra list; no spaces after commas. Paths must not contain commas.
  TRAIN_FILES_ARG="data.train_files=[${SWE_TRAIN_PARQUET},${SWE_MIX_PARQUET}]"
else
  TRAIN_FILES_ARG="data.train_files=${SWE_TRAIN_PARQUET}"
fi

# GLOBAL_TRAIN_BATCH_SIZE must be divisible by (NUM_GPUS / SP_SIZE). See verl FSDPSFTTrainer._normalize_config_bsz.
DP_GROUPS=$(( NUM_GPUS / SP_SIZE ))
if (( GLOBAL_TRAIN_BATCH_SIZE % DP_GROUPS != 0 )); then
  echo "Adjusting GLOBAL_TRAIN_BATCH_SIZE (${GLOBAL_TRAIN_BATCH_SIZE}) to be divisible by DP groups (${DP_GROUPS})"
  GLOBAL_TRAIN_BATCH_SIZE=$(( (GLOBAL_TRAIN_BATCH_SIZE / DP_GROUPS) * DP_GROUPS ))
  if (( GLOBAL_TRAIN_BATCH_SIZE < DP_GROUPS )); then
    GLOBAL_TRAIN_BATCH_SIZE=${DP_GROUPS}
  fi
fi

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  -m verl.trainer.fsdp_sft_trainer \
  "${TRAIN_FILES_ARG}" \
  data.val_files="${SWE_VAL_PARQUET}" \
  data.multiturn.enable=True \
  data.multiturn.messages_key="${MESSAGES_KEY}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation="${TRUNCATION}" \
  data.train_batch_size="${GLOBAL_TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  model.partial_pretrain="${SWE_MODEL_PATH}" \
  model.trust_remote_code="${TRUST_REMOTE_CODE}" \
  model.enable_gradient_checkpointing="${ENABLE_GRAD_CKPT}" \
  model.lora_rank="${LORA_RANK}" \
  optim.lr="${LEARNING_RATE}" \
  optim.warmup_steps_ratio="${WARMUP_RATIO}" \
  ulysses_sequence_parallel_size="${SP_SIZE}" \
  use_remove_padding="${USE_REMOVE_PADDING}" \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.project_name=swe-multiturn-sft \
  trainer.experiment_name=qwen32b-swe \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.logger="['console']" \
  ${HYDRA_EXTRA} \
  "$@"

echo "Done. Checkpoints under: ${SAVE_DIR} (global_step_*)"
