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
# Local directory: all *.parquet in the folder (sorted; see SWE_PARQUET_RECURSIVE):
#   export SWE_TRAIN_PARQUET_DIR=/data/swe_shards
#   export SWE_VAL_PARQUET=/data/val.parquet   # or SWE_VAL_PARQUET_DIR=... or SWE_REUSE_TRAIN_FOR_VAL=1
# Local explicit list (comma-separated paths — no spaces after commas):
#   export SWE_TRAIN_PARQUETS="/data/a.parquet,/data/b.parquet"
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
# Train: SWE_TRAIN_PARQUET_DIR (all *.parquet in dir) OR SWE_TRAIN_PARQUETS OR SWE_TRAIN_PARQUET (mutually exclusive base).
SWE_TRAIN_PARQUET_DIR="${SWE_TRAIN_PARQUET_DIR:-}"
SWE_TRAIN_PARQUET="${SWE_TRAIN_PARQUET:-}"
SWE_TRAIN_PARQUETS="${SWE_TRAIN_PARQUETS:-}"
SWE_VAL_PARQUET_DIR="${SWE_VAL_PARQUET_DIR:-}"
SWE_VAL_PARQUET="${SWE_VAL_PARQUET:-}"
SWE_VAL_PARQUETS="${SWE_VAL_PARQUETS:-}"
# If 1: use find; else only *.parquet directly under the train/val dir (maxdepth 1).
SWE_PARQUET_RECURSIVE="${SWE_PARQUET_RECURSIVE:-0}"
# If 1: use the same file list for val as for train (val loss is not held-out; see README).
SWE_REUSE_TRAIN_FOR_VAL="${SWE_REUSE_TRAIN_FOR_VAL:-0}"

# Optional: extra local parquet merged onto the training list (ignored if SWE_HF_DATASET_REPO is set).
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

# prefix: data.train_files or data.val_files; csv: comma-separated paths (no commas inside paths)
csv_to_hydra_files() {
  local prefix="$1"
  local csv="$2"
  local -a paths=()
  local p
  IFS=',' read -r -a _raw <<< "${csv}"
  for p in "${_raw[@]}"; do
    [[ -z "${p}" ]] && continue
    paths+=("${p}")
  done
  local n=${#paths[@]}
  if [[ "${n}" -eq 0 ]]; then
    echo ""
    return 1
  elif [[ "${n}" -eq 1 ]]; then
    echo "${prefix}=${paths[0]}"
  else
    local out="${prefix}=[${paths[0]}"
    local i=1
    while [[ "${i}" -lt "${n}" ]]; do
      out+=",${paths[i]}"
      i=$((i + 1))
    done
    out+="]"
    echo "${out}"
  fi
}

verify_parquet_csv() {
  local csv="$1"
  local label="$2"
  local p
  IFS=',' read -r -a _raw <<< "${csv}"
  for p in "${_raw[@]}"; do
    [[ -z "${p}" ]] && continue
    if [[ ! -f "${p}" ]]; then
      echo "ERROR: ${label} parquet not found: ${p}" >&2
      exit 1
    fi
  done
}

# Print comma-separated absolute paths of *.parquet under dir (sorted). $2 = recursive (1/true = subfolders).
discover_parquet_dir_csv() {
  local dir="$1"
  local recursive="${2:-0}"
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: Parquet directory does not exist or is not a directory: ${dir}" >&2
    exit 1
  fi
  local -a files=()
  local f
  if [[ "${recursive}" == "1" || "${recursive}" == "true" ]]; then
    while IFS= read -r f; do
      [[ -n "${f}" ]] && files+=("${f}")
    done < <(find "${dir}" -type f \( -name '*.parquet' -o -name '*.PARQUET' \) | LC_ALL=C sort)
  else
    while IFS= read -r f; do
      [[ -n "${f}" ]] && files+=("${f}")
    done < <(find "${dir}" -maxdepth 1 -type f \( -name '*.parquet' -o -name '*.PARQUET' \) | LC_ALL=C sort)
  fi
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: No .parquet files under ${dir} (recursive=${recursive})" >&2
    exit 1
  fi
  local i out=""
  for ((i = 0; i < ${#files[@]}; i++)); do
    [[ ${i} -gt 0 ]] && out+=","
    out+="${files[i]}"
  done
  echo "${out}"
}

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
  _train_sources=0
  [[ -n "${SWE_TRAIN_PARQUET_DIR}" ]] && _train_sources=$((_train_sources + 1))
  [[ -n "${SWE_TRAIN_PARQUETS}" ]] && _train_sources=$((_train_sources + 1))
  [[ -n "${SWE_TRAIN_PARQUET}" ]] && _train_sources=$((_train_sources + 1))
  if [[ "${_train_sources}" -gt 1 ]]; then
    echo "ERROR: Use only one of SWE_TRAIN_PARQUET_DIR, SWE_TRAIN_PARQUETS, or SWE_TRAIN_PARQUET (then optional SWE_MIX_PARQUET)." >&2
    exit 1
  fi

  TRAIN_CSV=""
  if [[ -n "${SWE_TRAIN_PARQUET_DIR}" ]]; then
    TRAIN_CSV="$(discover_parquet_dir_csv "${SWE_TRAIN_PARQUET_DIR}" "${SWE_PARQUET_RECURSIVE}")"
  elif [[ -n "${SWE_TRAIN_PARQUETS}" ]]; then
    TRAIN_CSV="${SWE_TRAIN_PARQUETS}"
  elif [[ -n "${SWE_TRAIN_PARQUET}" ]]; then
    TRAIN_CSV="${SWE_TRAIN_PARQUET}"
  else
    echo "ERROR: Set SWE_TRAIN_PARQUET_DIR, SWE_TRAIN_PARQUETS, or SWE_TRAIN_PARQUET, or use SWE_HF_DATASET_REPO." >&2
    exit 1
  fi
  if [[ -n "${SWE_MIX_PARQUET}" ]]; then
    if [[ ! -f "${SWE_MIX_PARQUET}" ]]; then
      echo "ERROR: SWE_MIX_PARQUET set but file missing: ${SWE_MIX_PARQUET}" >&2
      exit 1
    fi
    TRAIN_CSV="${TRAIN_CSV},${SWE_MIX_PARQUET}"
  fi

  verify_parquet_csv "${TRAIN_CSV}" "Train"
  TRAIN_FILES_ARG="$(csv_to_hydra_files "data.train_files" "${TRAIN_CSV}")"

  _val_sources=0
  [[ "${SWE_REUSE_TRAIN_FOR_VAL}" == "1" || "${SWE_REUSE_TRAIN_FOR_VAL}" == "true" ]] && _val_sources=$((_val_sources + 1))
  [[ -n "${SWE_VAL_PARQUET_DIR}" ]] && _val_sources=$((_val_sources + 1))
  [[ -n "${SWE_VAL_PARQUETS}" ]] && _val_sources=$((_val_sources + 1))
  [[ -n "${SWE_VAL_PARQUET}" ]] && _val_sources=$((_val_sources + 1))
  if [[ "${_val_sources}" -gt 1 ]]; then
    echo "ERROR: Use only one of SWE_REUSE_TRAIN_FOR_VAL, SWE_VAL_PARQUET_DIR, SWE_VAL_PARQUETS, or SWE_VAL_PARQUET." >&2
    exit 1
  fi

  VAL_CSV=""
  if [[ "${SWE_REUSE_TRAIN_FOR_VAL}" == "1" || "${SWE_REUSE_TRAIN_FOR_VAL}" == "true" ]]; then
    VAL_CSV="${TRAIN_CSV}"
    echo "WARN: SWE_REUSE_TRAIN_FOR_VAL: validation uses the same parquet file(s) as training; val loss is not a clean held-out metric." >&2
  elif [[ -n "${SWE_VAL_PARQUET_DIR}" ]]; then
    VAL_CSV="$(discover_parquet_dir_csv "${SWE_VAL_PARQUET_DIR}" "${SWE_PARQUET_RECURSIVE}")"
  elif [[ -n "${SWE_VAL_PARQUETS}" ]]; then
    VAL_CSV="${SWE_VAL_PARQUETS}"
  elif [[ -n "${SWE_VAL_PARQUET}" ]]; then
    VAL_CSV="${SWE_VAL_PARQUET}"
  else
    echo "ERROR: No validation data. Set SWE_VAL_PARQUET_DIR, SWE_VAL_PARQUET, SWE_VAL_PARQUETS, or SWE_REUSE_TRAIN_FOR_VAL=1 (see recipe/swe/README.md)." >&2
    exit 1
  fi

  verify_parquet_csv "${VAL_CSV}" "Validation"
  VAL_FILES_ARG="$(csv_to_hydra_files "data.val_files" "${VAL_CSV}")"
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
