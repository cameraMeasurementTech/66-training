# KlearReasoner — Project overview

This document describes what the **KlearReasoner** repository is, how it is organized, and how training and evaluation fit together. It complements the root [README.md](../README.md).

---

## 1. Purpose

**KlearReasoner** is a research and training codebase built around an **8B-parameter reasoning LLM** (“Klear-Reasoner-8B”) aimed at strong **math** and **code** performance with **long chain-of-thought (CoT)** behavior.

The main scientific contribution is **GPPO (Gradient-Preserving Clipping Policy Optimization)**: a modification to clipped policy-gradient RL that **still backpropagates through clipped tokens**, with controlled gradient size, to mitigate two issues of vanilla PPO/GRPO-style clipping:

- **Exploration**: high-entropy tokens with large importance ratios are often fully dropped from the graph.
- **Correction**: suboptimal trajectories (small ratio, negative advantage) can be ignored too aggressively.

The README also points to follow-on work (**CE-GPPO**, **entropy ratio clipping / ERC**) with separate papers.

**Primary references**

- [Klear-Reasoner paper (arXiv:2508.07629)](https://arxiv.org/pdf/2508.07629)
- [CE-GPPO (arXiv:2509.20712)](https://arxiv.org/abs/2509.20712)
- [ERC (arXiv:2512.05591)](https://arxiv.org/abs/2512.05591)

---

## 2. Published model and results

Training stages (from the README):

1. **Quality-centric long CoT SFT** — distilled from **DeepSeek-R1-0528**.
2. **RL with GPPO** — on math/code-style data with rule-based verifiers.

Reported benchmarks include **AIME 2024/2025**, **LiveCodeBench V5/V6**, etc., with **pass@1**-style metrics. Strong results are reported with an extended inference setup (e.g. **64K budget** and **YaRN** scaling); see the README tables for exact numbers.

**Artifacts**: Hugging Face model and datasets are linked from the root README (math RL subset ~30K, code RL subset ~15K).

---

## 3. Repository structure (software)

This repository is a **full RL training stack** based on **verl** (“Volcano Engine Reinforcement Learning for LLM” per `pyproject.toml`): distributed training, FSDP actors, **vLLM** (and optional **SGLang**) rollouts, Ray orchestration, Hydra configs, and DAPO-style reward/filtering.

| Area | Role |
|------|------|
| **`verl/`** | Core library: PPO/GRPO/GPPO losses, Ray trainer, workers (actor, critic, rollout, reward), checkpointing, utilities, reward scorers for many tasks. |
| **`recipe/dapo/`** | **KlearReasoner training entrypoint** — shell scripts that `ray job submit` → `python -m recipe.dapo.src.main_dapo` with Hydra overrides. |
| **`benchmarks/`** | **Inference** (`inference.py` via vLLM) and **math judging** (`judge_math.py` using `math_verify`). |
| **`examples/`**, **`docs/`** | Additional verl examples and Sphinx documentation. |

**Licensing note:** `Notice.txt` attributes copyright to **Bytedance** for the verl base; the root README describes the KlearReasoner research project.

---

## 4. GPPO in code

**Policy losses** live in `verl/trainer/ppo/core_algos.py`:

- **`compute_policy_loss`** — standard clipped loss (used when `loss_func == "grpo"`).
- **`compute_gppo_loss`** — GPPO: the clipped surrogate uses terms like `(1 ± ε) / ratio.detach() * ratio` so the **forward** behavior aligns with clip-higher-style PPO while **gradients** still flow through `ratio`. Supports an **`only_high`** option (from actor config) for variants that apply the gradient-preserving trick only on the **upper** clip side.
- **`compute_gppo_loss_general_beta`** — tunable **β₁, β₂** scaling for low/high clip regions (`loss_func == "gppo_general"`), for cases where the low-side gradient is too strong (e.g. entropy collapse).

The **actor** selects the loss in `verl/workers/actor/dp_actor.py` via `loss_func`: `"grpo"`, `"gppo"`, or `"gppo_general"`.

Defaults are in `verl/trainer/config/ppo_trainer.yaml` (e.g. **`loss_func: "gppo"`**, `clip_ratio_low` / `clip_ratio_high`, dual-clip `clip_ratio_c`, optional `positive_loss_coeff`, KL-related options for GRPO-style runs).

---

## 5. Training pipeline

1. **Environment**: `pip install -e .` and `pip install -r requirements.txt` (see root README). Dependencies include **Ray**, **vLLM**, **Hydra**, **math-verify**, **Pebble**, and (for code) **Firejail** sandboxing as described in the README.
2. **Orchestration**: Scripts such as `recipe/dapo/perf_run_dapo_ours_math.sh` and `perf_run_dapo_ours_code.sh` configure NCCL, model path, checkpoint directory, train/val data paths, batch sizes, sequence lengths (often **32K** response length for math), **GRPO** as advantage estimator (`adv_estimator=grpo`), asymmetric clip ranges, **DAPO** reward manager, and multi-node settings (`NNODES`).
3. **Execution**: `ray job submit` runs **`recipe.dapo.src.main_dapo`**, which uses **`RayDAPOTrainer`** (`recipe/dapo/src/dapo_ray_trainer.py`) on a Ray cluster.
4. **Configs**: `recipe/dapo/src/config/dapo_trainer.yaml` composes **`verl/trainer/config/ppo_trainer.yaml`** via Hydra `searchpath`.
5. **Rewards**: `verl/workers/reward_manager/dapo.py` implements **`DAPORewardManager`**, using **`dapo_wi_rllm_compute_score`** (and related scoring config) with **Pebble** process pools, timeouts, and optional **overlong** buffer / filtering. The **`data_source`** field in each training example selects the verifier (see README).
6. **Stability**: The README recommends **`actor_rollout_ref.actor.overlong_filter=True`** when max sequence length is **below 32K**.

**Variant scripts** (see README):

- `recipe/dapo/perf_run_dapo_ours_math_general_gppo.sh` — β-scaled GPPO (`gppo_general`).
- `recipe/dapo/perf_run_dapo_ours_math_only_high.sh` — high-side-only GPPO gradient variant.

---

## 6. RL data format

**Math** (illustrative JSON fields from the README):

- `prompt` (chat messages), `ability: "math"`, `data_source: "math_longcot_math_verify"`, `reward_model.ground_truth`, `style: "rule"`.

**Code**:

- `ability: "code"`, `data_source: "coder1_longcot"`, ground truth / tests for rule-based scoring.

Training code keys off **`data_source`** and **`reward_fn_key`** (default `data_source` in base config).

---

## 7. Evaluation and inference

- **`benchmarks/inference.py`**: Loads a dataset, repeats each prompt **n** times (e.g. **64** for avg@64), runs **vLLM** with settings such as **YaRN**-style `rope_scaling`, large context limits, **temperature 0.6**, **top_p 0.95**.
- **`benchmarks/judge_math.py`**: Scores outputs with **`math_verify`** (`parse` / `verify`), including stripping of thinking blocks and comparison to reference solutions.

Example commands are in the root README under **Evaluation**.

---

## 8. Multi-node Ray (summary)

1. Head node: `ray start --head --dashboard-host=0.0.0.0`
2. Workers: `ray start --address="<HEAD_IP>:6379"`
3. Submit the training job from the master node using the `recipe/dapo/*.sh` scripts.

Details are in the root README **Using Ray for Multi-Node Training** section.

---

## 9. One-line summary

**KlearReasoner** provides a **long-CoT 8B reasoning model recipe**, a **GPPO-based RL implementation** inside a **verl-based training stack**, and **scripts plus benchmarks** to reproduce math/code RL training and evaluation.

---

## Citation

Use the BibTeX entries in the root [README.md](../README.md) for the KlearReasoner, CE-GPPO, and ERC papers.
