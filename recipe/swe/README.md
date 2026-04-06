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

You need **at least one training Parquet**. The trainer also expects a **validation** set; you can use a directory of val shards, a single file, comma-separated paths, or temporarily reuse training data (see below).

**Directory of training shards (simplest when everything is in one folder)** — the script collects every **`*.parquet`** in that directory, **sorted by path** (default: only the folder itself, not subfolders):

```bash
export SWE_TRAIN_PARQUET_DIR="/data/swe_parquets"
export SWE_VAL_PARQUET="/data/val.parquet"   # or SWE_VAL_PARQUET_DIR, SWE_VAL_PARQUETS, or SWE_REUSE_TRAIN_FOR_VAL=1
```

Include **subdirectories** in the scan:

```bash
export SWE_PARQUET_RECURSIVE=1
```

**Directory of validation Parquet files** — same rules:

```bash
export SWE_VAL_PARQUET_DIR="/data/val_shards"
```

Do **not** combine `SWE_TRAIN_PARQUET_DIR` with `SWE_TRAIN_PARQUETS` or `SWE_TRAIN_PARQUET` (pick one base source). You can still append **`SWE_MIX_PARQUET`** when using a train directory.

**Single train + single val file** — set `SWE_TRAIN_PARQUET` and `SWE_VAL_PARQUET`.

**Several training shards as explicit paths** — comma-separated (**no spaces** after commas):

```bash
export SWE_TRAIN_PARQUETS="/data/part-000.parquet,/data/part-001.parquet,/data/part-002.parquet"
```

Then choose validation **one** of these ways:

1. **Held-out (recommended)** — reserve one shard only for metrics, e.g.  
   `export SWE_VAL_PARQUET="/data/part-002.parquet"` and remove that path from `SWE_TRAIN_PARQUETS` / exclude it from `SWE_TRAIN_PARQUET_DIR`, **or** use a small `val.parquet` you built from a random slice of rows.
2. **Quick / debugging** — the trainer still requires `val_files`; you can point validation at the **same** list as training:  
   `export SWE_REUSE_TRAIN_FOR_VAL=1`  
   Val loss is then **not** a clean held-out metric; use training loss and checkpoints with care.

**Several validation Parquet files as paths** — comma-separated: `SWE_VAL_PARQUETS="/data/v0.parquet,/data/v1.parquet"`.

Use **`SWE_TRAIN_PARQUET`** when you only have a single training file and no directory to scan.

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

**Train directory + single val file** (typical for many shards in one folder):

```bash
export SWE_MODEL_PATH="/path/to/Qwen-32B-Instruct"
export SWE_TRAIN_PARQUET_DIR="/data/swe_train_parquets"
export SWE_VAL_PARQUET="/data/val.parquet"
export NUM_GPUS=8

./recipe/swe/run_swe_multiturn_sft.sh
```

**Train directory only, reuse train for val** (debugging):

```bash
export SWE_MODEL_PATH="/path/to/Qwen-32B-Instruct"
export SWE_TRAIN_PARQUET_DIR="/data/swe_train_parquets"
export SWE_REUSE_TRAIN_FOR_VAL=1
export NUM_GPUS=8
./recipe/swe/run_swe_multiturn_sft.sh
```

**One train file and one val file:**

```bash
export SWE_MODEL_PATH="/path/to/Qwen-32B-Instruct"
export SWE_TRAIN_PARQUET="/path/to/train.parquet"
export SWE_VAL_PARQUET="/path/to/val.parquet"
export NUM_GPUS=8

# Optional: append another parquet to the training list (local mode only)
# export SWE_MIX_PARQUET="/path/to/general.parquet"

./recipe/swe/run_swe_multiturn_sft.sh
```

**Explicit list of training Parquet paths** (when not using a directory):

```bash
export SWE_MODEL_PATH="/path/to/Qwen-32B-Instruct"
export SWE_TRAIN_PARQUETS="/data/shard_00.parquet,/data/shard_01.parquet,/data/shard_02.parquet"
export SWE_REUSE_TRAIN_FOR_VAL=1   # or set SWE_VAL_PARQUET / SWE_VAL_PARQUET_DIR
export NUM_GPUS=8

./recipe/swe/run_swe_multiturn_sft.sh
```

Use **exactly one** of `SWE_TRAIN_PARQUET_DIR`, `SWE_TRAIN_PARQUETS`, or `SWE_TRAIN_PARQUET` as the training source. `SWE_MIX_PARQUET` may still be appended to that list.

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
| `SWE_TRAIN_PARQUET_DIR` | Directory: all `*.parquet` inside (sorted); use **one** of this, `SWE_TRAIN_PARQUETS`, or `SWE_TRAIN_PARQUET` |
| `SWE_PARQUET_RECURSIVE` | If `1` or `true`, include `*.parquet` in subfolders when using `*_PARQUET_DIR` |
| `SWE_TRAIN_PARQUET` | Single local training Parquet |
| `SWE_TRAIN_PARQUETS` | Comma-separated training Parquet paths |
| `SWE_VAL_PARQUET_DIR` | Directory: all `*.parquet` used as validation (sorted) |
| `SWE_VAL_PARQUET` | Single local validation Parquet |
| `SWE_VAL_PARQUETS` | Comma-separated validation Parquet paths |
| `SWE_REUSE_TRAIN_FOR_VAL` | If `1` or `true`, reuse the **training** file list for validation (not a held-out metric) |
| `SWE_MIX_PARQUET` | Extra local Parquet appended to the training list (local data mode only) |
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
| Empty `SWE_*_PARQUET_DIR` | Parquet files must be in that directory (or set `SWE_PARQUET_RECURSIVE=1` if they live in subfolders). |
| Wrong chat format | Align `messages` / roles with what the base model’s `chat_template` expects. |
| `flash_attention_2` errors | Install a compatible FlashAttention build for your PyTorch/CUDA stack. |

---

## 10. Files in this recipe

| File | Role |
|------|------|
| `run_swe_multiturn_sft.sh` | Main launcher (`torchrun` + Hydra overrides) |
| `hf_fetch_parquet_for_sft.py` | Resolve/download Hub Parquet paths |
| `jsonl_to_multiturn_parquet.py` | JSONL → Parquet for `messages` |
