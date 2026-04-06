#!/usr/bin/env bash
# Multi-turn supervised fine-tuning for SWE-style trajectories on top of verl's FSDP SFT trainer.
# Uses the same tokenizer and vocabulary as the base checkpoint (model.partial_pretrain).
# Full model fine-tune: keep LORA_RANK=0 (trains all parameters; model size unchanged).
#
# --- Model (pick one) ---
# Local directory with config.json:
#   export SWE_MODEL_PATH=/path/to/Qwen-32B-Instruct
# Hugging Face Hub model id (verl uses from_pretrained; weights stream from cache):
#   export SWE_MODEL_PATH=Qwen/Qwen2.5-32B-Instruct
#
# --- Data (pick one) ---
# Local parquet:
#   export SWE_TRAIN_PARQUET=... SWE_VAL_PARQUET=...
# Hub dataset repo (parquet files in the repo; uses huggingface_hub to download/cache):
#   export SWE_HF_DATASET_REPO=org/my-dataset
#   export SWE_HF_TRAIN_GLOBS='data/train-*.parquet'   # comma-separated for multiple globs
#   export SWE_HF_VAL_GLOBS='data/validation-*.parquet'  # optional if SWE_VAL_PARQUET is set instead
#
# Gated assets: export HF_TOKEN=... (or huggingface-cli login).
#
# Mitigate forgetting: mix data (local SWE_MIX_PARQUET), moderate LR, few epochs — not combinable with
# SWE_HF_DATASET_REPO in this script; use HYDRA_EXTRA to pass extra train_files if needed.
#
# After SFT, RL + GPPO: recipe/dapo/*.sh with actor_rollout_ref.model.path pointing at the SFT checkpoint.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- Model (local dir with config.json, or Hub id e.g. Qwen/Qwen2.5-32B-Instruct) ----
SWE_MODEL_PATH="${SWE_MODEL_PATH:-/path/to/your/Qwen-32B-Instruct}"

# ---- Local data (used when SWE_HF_DATASET_REPO is empty) ----
SWE_TRAIN_PARQUET="${SWE_TRAIN_PARQUET:-/path/to/swe_train.parquet}"
SWE_VAL_PARQUET="${SWE_VAL_PARQUET:-/path/to/swe_val.parquet}"

# Optional: second local parquet merged into training (ignored if SWE_HF_DATASET_REPO is set).
SWE_MIX_PARQUET="${SWE_MIX_PARQUET:-}"

# ---- Hugging Face dataset (parquet files in repo) ----
SWE_HF_DATASET_REPO="${SWE_HF_DATASET_REPO:-}"
SWE_HF_REVISION="${SWE_HF_REVISION:-}"
# Comma-separated globs vs paths in the repo, e.g. data/train-*.parquet,splits/train/*.parquet
SWE_HF_TRAIN_GLOBS="${SWE_HF_TRAIN_GLOBS:-}"
SWE_HF_VAL_GLOBS="${SWE_HF_VAL_GLOBS:-}"

# ---- Cluster ----
NUM_GPUS="${NUM_GPUS:-8}"
SP_SIZE="${SP_SIZE:-1}"

MESSAGES_KEY="${MESSAGES_KEY:-messages}"

GLOBAL_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE:-${NUM_GPUS}}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
TRUNCATION="${TRUNCATION:-left}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_DIR="${SAVE_DIR:-${HOME}/ckpts/swe-qwen32b-multiturn-sft}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
LORA_RANK="${LORA_RANK:-0}"

HYDRA_EXTRA="${HYDRA_EXTRA:-}"

# --- Validate model path: local checkpoint dir OR Hub id ---
if [[ -d "${SWE_MODEL_PATH}" ]]; then
  if [[ ! -f "${SWE_MODEL_PATH}/config.json" ]]; then
    echo "ERROR: SWE_MODEL_PATH is a directory but missing config.json: ${SWE_MODEL_PATH}" >&2
    exit 1
  fi
fi

# --- Resolve data paths ---
if [[ -n "${SWE_HF_DATASET_REPO}" ]]; then
  if [[ -n "${SWE_MIX_PARQUET}" ]]; then
    echo "ERROR: Do not set SWE_MIX_PARQUET together with SWE_HF_DATASET_REPO (extend globs or use HYDRA_EXTRA)." >&2
    exit 1
  fi
  if [[ -z "${SWE_HF_TRAIN_GLOBS}" ]]; then
    echo "ERROR: SWE_HF_DATASET_REPO is set but SWE_HF_TRAIN_GLOBS is empty (comma-separated fnmatch globs)." >&2
    exit 1
  fi

  HF_FETCH_ARGS=(
    "${REPO_ROOT}/recipe/swe/hf_fetch_parquet_for_sft.py"
    --repo-id "${SWE_HF_DATASET_REPO}"
    --emit hydra
  )
  if [[ -n "${SWE_HF_REVISION}" ]]; then
    HF_FETCH_ARGS+=(--revision "${SWE_HF_REVISION}")
  fi
  IFS=',' read -r -a _train_globs <<< "${SWE_HF_TRAIN_GLOBS}"
  for _g in "${_train_globs[@]}"; do
    [[ -z "${_g}" ]] && continue
    HF_FETCH_ARGS+=(--train-glob "${_g}")
  done
  if [[ -n "${SWE_HF_VAL_GLOBS}" ]]; then
    IFS=',' read -r -a _val_globs <<< "${SWE_HF_VAL_GLOBS}"
    for _g in "${_val_globs[@]}"; do
      [[ -z "${_g}" ]] && continue
      HF_FETCH_ARGS+=(--val-glob "${_g}")
    done
  fi

  # shellcheck disable=SC2312
  _py=python3
  command -v python3 >/dev/null 2>&1 || _py=python
  eval "$("${_py}" "${HF_FETCH_ARGS[@]}")"

  if [[ -z "${TRAIN_FILES_ARG:-}" ]]; then
    echo "ERROR: hf_fetch_parquet_for_sft.py did not set TRAIN_FILES_ARG" >&2
    exit 1
  fi
  if [[ -z "${VAL_FILES_ARG:-}" ]]; then
    if [[ ! -f "${SWE_VAL_PARQUET}" ]]; then
      echo "ERROR: Hub run did not match val parquet; set SWE_HF_VAL_GLOBS or a local SWE_VAL_PARQUET file." >&2
      exit 1
    fi
    VAL_FILES_ARG="data.val_files=${SWE_VAL_PARQUET}"
  fi
else
  if [[ ! -f "${SWE_TRAIN_PARQUET}" ]]; then
    echo "ERROR: SWE_TRAIN_PARQUET not found: ${SWE_TRAIN_PARQUET} (or set SWE_HF_DATASET_REPO + SWE_HF_TRAIN_GLOBS)" >&2
    exit 1
  fi
  if [[ ! -f "${SWE_VAL_PARQUET}" ]]; then
    echo "ERROR: SWE_VAL_PARQUET not found: ${SWE_VAL_PARQUET}" >&2
    exit 1
  fi

  if [[ -n "${SWE_MIX_PARQUET}" ]]; then
    if [[ ! -f "${SWE_MIX_PARQUET}" ]]; then
      echo "ERROR: SWE_MIX_PARQUET set but file missing: ${SWE_MIX_PARQUET}" >&2
      exit 1
    fi
    TRAIN_FILES_ARG="data.train_files=[${SWE_TRAIN_PARQUET},${SWE_MIX_PARQUET}]"
  else
    TRAIN_FILES_ARG="data.train_files=${SWE_TRAIN_PARQUET}"
  fi
  VAL_FILES_ARG="data.val_files=${SWE_VAL_PARQUET}"
fi

mkdir -p "${SAVE_DIR}"

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
  "${VAL_FILES_ARG}" \
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
