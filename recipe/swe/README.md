# SWE multi-turn supervised fine-tuning (SFT)

This recipe fine-tunes a causal LM on **multi-turn chat trajectories** (e.g. software-engineering agent traces) using verl’s **FSDP SFT trainer** (`verl.trainer.fsdp_sft_trainer`) and **`MultiTurnSFTDataset`**. Training loss is applied only on **assistant** tokens; the tokenizer is loaded from the **same checkpoint** as the model, so vocabulary and tokenization stay aligned with the base model.

---

## 1. Prerequisites

- **Linux**, **NVIDIA GPUs** with enough aggregate memory for your model (32B full fine-tune typically needs a multi-GPU node; tune batch size and sequence length).
- **Python 3.8+**, **PyTorch** with CUDA, **FlashAttention 2** (as required by the stock SFT trainer).
- This repository installed in editable mode from the **KlearReasoner root**:

```bash
cd /path/to/KlearReasoner
pip install -e .
pip install -r requirements.txt
```

- For **Hugging Face** models or datasets: `huggingface_hub` (usually installed with `transformers` / `datasets`). For gated assets, run `huggingface-cli login` or set `HF_TOKEN`.

---

## 2. Data format

Training and validation data must be **Parquet** with a column of **message lists** (name configurable; default **`messages`**). Each row is one conversation:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Roles must include **`assistant`** where you want the loss. The template applied is the model’s **`chat_template`** (from the tokenizer), so keep the same conventions as your base instruct model (e.g. Qwen).

**Long sequences:** The script defaults to a large `MAX_LENGTH` and `truncation=left`. If you hit OOM, lower `MAX_LENGTH`, increase `SP_SIZE` (sequence parallel), or use a smaller global batch.

---

## 3. Prepare data (from scratch)

### Option A — Already Parquet locally

Ensure two files exist, for example:

- `train.parquet` — training split  
- `val.parquet` — validation split  

### Option B — JSONL → Parquet

If you have one JSON object per line, each with a `messages` field:

```bash
cd /path/to/KlearReasoner
python3 recipe/swe/jsonl_to_multiturn_parquet.py \
  /path/to/train.jsonl \
  /path/to/train.parquet

python3 recipe/swe/jsonl_to_multiturn_parquet.py \
  /path/to/val.jsonl \
  /path/to/val.parquet
```

Optional: `--messages-key your_column` if the field is not named `messages`.

### Option C — Parquet on the Hugging Face Hub

You do **not** need to download shards manually for training if you use the env vars in §5.2. To **inspect** which files and paths match your globs:

```bash
python3 recipe/swe/hf_fetch_parquet_for_sft.py \
  --repo-id your-org/your-dataset \
  --train-glob 'data/train-*.parquet' \
  --val-glob 'data/validation-*.parquet' \
  --emit json
```

Globs are **fnmatch** patterns against **repository-relative paths** (and basenames). Use the dataset’s **Files** view on the Hub to pick correct patterns.

---

## 4. Prepare the base model

### Local snapshot

Download or copy a Hugging Face–style directory that contains **`config.json`**, weights, and tokenizer files. Point `SWE_MODEL_PATH` at that directory.

### Hugging Face Hub id

Use the model id string (e.g. `Qwen/Qwen2.5-32B-Instruct`) as `SWE_MODEL_PATH`. The trainer uses `from_pretrained`; weights are cached under the Hugging Face cache on first use.

---

## 5. Run training

Always run the launcher from the **KlearReasoner repository root** (the script `cd`s there automatically if you invoke it by path).

```bash
cd /path/to/KlearReasoner
chmod +x recipe/swe/run_swe_multiturn_sft.sh   # once
```

### 5.1 Local model + local Parquet

```bash
export SWE_MODEL_PATH="/path/to/Qwen-32B-Instruct"
export SWE_TRAIN_PARQUET="/path/to/train.parquet"
export SWE_VAL_PARQUET="/path/to/val.parquet"
export NUM_GPUS=8

# Optional: mix general instruct data to reduce forgetting (second parquet concatenated in training)
# export SWE_MIX_PARQUET="/path/to/general.parquet"

./recipe/swe/run_swe_multiturn_sft.sh
```

### 5.2 Hugging Face model + Hub Parquet

```bash
export SWE_MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
export SWE_HF_DATASET_REPO="your-org/your-swe-dataset"
export SWE_HF_TRAIN_GLOBS="data/train-*.parquet"
export SWE_HF_VAL_GLOBS="data/validation-*.parquet"
export NUM_GPUS=8

# Optional pin:
# export SWE_HF_REVISION="main"

./recipe/swe/run_swe_multiturn_sft.sh
```

If you **omit** `SWE_HF_VAL_GLOBS`, you must supply a **local** validation file:

```bash
export SWE_VAL_PARQUET="/path/to/val.parquet"
```

**Note:** `SWE_MIX_PARQUET` cannot be used together with `SWE_HF_DATASET_REPO` in this script. Add more train globs or pass extra Hydra overrides (see §6).

### 5.3 Useful environment variables

| Variable | Meaning |
|----------|--------|
| `SWE_MODEL_PATH` | Local model dir (with `config.json`) or Hub model id |
| `SWE_TRAIN_PARQUET` / `SWE_VAL_PARQUET` | Local Parquet paths (Hub data mode uses train from Hub; val optional) |
| `SWE_MIX_PARQUET` | Extra local train Parquet (local data mode only) |
| `SWE_HF_DATASET_REPO` | Hub dataset repo id |
| `SWE_HF_REVISION` | Optional revision |
| `SWE_HF_TRAIN_GLOBS` | Comma-separated globs for train `.parquet` files in the repo |
| `SWE_HF_VAL_GLOBS` | Comma-separated globs for val `.parquet` files |
| `NUM_GPUS` | `torchrun` `--nproc_per_node` |
| `SP_SIZE` | Ulysses sequence-parallel size (try `2` or `4` for long context) |
| `GLOBAL_TRAIN_BATCH_SIZE` | Global batch before internal DP normalization (must be divisible by data-parallel groups) |
| `MICRO_BATCH_SIZE_PER_GPU` | Micro-batch for gradient accumulation |
| `MAX_LENGTH` | Max sequence length (tokens) |
| `TRUNCATION` | `left`, `right`, or `error` |
| `LEARNING_RATE` | Default `1e-5` |
| `TOTAL_EPOCHS` | Default `1` |
| `SAVE_DIR` | Checkpoint root |
| `LORA_RANK` | `0` = full fine-tune (default); set `>0` for LoRA |
| `TRUST_REMOTE_CODE` | `True` for many Qwen checkpoints |
| `HYDRA_EXTRA` | Extra space-separated Hydra overrides |

---

## 6. Extra Hydra overrides

Append arguments **after** the script, or use `HYDRA_EXTRA`:

```bash
HYDRA_EXTRA="trainer.logger=['console','wandb'] optim.weight_decay=0" \
  ./recipe/swe/run_swe_multiturn_sft.sh \
  data.max_length=16384 \
  optim.lr=5e-6
```

Default config is merged from `verl/trainer/config/sft_trainer.yaml`.

---

## 7. Checkpoints

Checkpoints are written under `SAVE_DIR` (default `~/ckpts/swe-qwen32b-multiturn-sft`) as **`global_step_*`** directories in **Hugging Face** format (model + tokenizer).

---

## 8. After SFT

To continue with **RL and GPPO** in this codebase, start from the DAPO-style scripts under `recipe/dapo/` and set `actor_rollout_ref.model.path` to your SFT output directory. See the root [README.md](../../README.md) for GPPO and cluster setup (Ray, etc.).

---

## 9. Troubleshooting

| Issue | Suggestions |
|-------|-------------|
| CUDA OOM | Lower `MAX_LENGTH`, `GLOBAL_TRAIN_BATCH_SIZE`, or `MICRO_BATCH_SIZE_PER_GPU`; enable `SP_SIZE>1`; keep `ENABLE_GRAD_CKPT=True`. |
| Hub 401 / gated dataset | `huggingface-cli login` or `HF_TOKEN`. |
| No Parquet matched | Fix `SWE_HF_*_GLOBS` using `hf_fetch_parquet_for_sft.py --emit json` or the Hub file browser. |
| Wrong chat format | Align `messages` / roles with what the base model’s `chat_template` expects. |
| `flash_attention_2` errors | Install a compatible FlashAttention build for your PyTorch/CUDA stack. |

---

## 10. Files in this recipe

| File | Role |
|------|------|
| `run_swe_multiturn_sft.sh` | Main launcher (`torchrun` + Hydra overrides) |
| `hf_fetch_parquet_for_sft.py` | Resolve/download Hub Parquet paths |
| `jsonl_to_multiturn_parquet.py` | JSONL → Parquet for `messages` |
