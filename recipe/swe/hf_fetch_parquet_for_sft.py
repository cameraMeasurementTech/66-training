#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download Parquet files from a Hugging Face dataset repo and emit paths / Hydra snippets for SFT."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import sys
from typing import Iterable


def _collect_matches(repo_files: list[str], patterns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for rel in repo_files:
        if not rel.endswith(".parquet"):
            continue
        base = rel.split("/")[-1]
        for pat in patterns:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(base, pat):
                seen.add(rel)
                break
    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id, e.g. org/my-swe-dataset",
    )
    parser.add_argument("--revision", default=None, help="Git revision (branch name, tag, or commit)")
    parser.add_argument("--repo-type", default="dataset", choices=("dataset", "model", "space"))
    parser.add_argument(
        "--train-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob matched against repo-relative paths (repeatable), e.g. 'data/train-*.parquet'",
    )
    parser.add_argument(
        "--val-glob",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob for validation parquet files (repeatable)",
    )
    parser.add_argument(
        "--emit",
        choices=("json", "hydra", "paths"),
        default="json",
        help="json: one JSON object; hydra: shell-safe TRAIN_FILES_ARG / VAL_FILES_ARG; paths: train paths one per line then VAL: lines",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        print("Install huggingface_hub (e.g. pip install huggingface_hub)", file=sys.stderr)
        raise e

    api = HfApi()
    repo_files = api.list_repo_files(args.repo_id, repo_type=args.repo_type, revision=args.revision)

    if not args.train_glob:
        print("Provide at least one --train-glob", file=sys.stderr)
        sys.exit(2)

    train_rel = _collect_matches(repo_files, args.train_glob)
    if not train_rel:
        print(f"No train parquet matched {args.train_glob} in {args.repo_id}", file=sys.stderr)
        sys.exit(3)

    val_rel: list[str] = []
    if args.val_glob:
        val_rel = _collect_matches(repo_files, args.val_glob)
        if not val_rel:
            print(f"No val parquet matched {args.val_glob} in {args.repo_id}", file=sys.stderr)
            sys.exit(4)

    def download(rel: str) -> str:
        return hf_hub_download(
            repo_id=args.repo_id,
            filename=rel,
            repo_type=args.repo_type,
            revision=args.revision,
        )

    train_paths = [download(r) for r in train_rel]
    val_paths = [download(r) for r in val_rel]

    if args.emit == "json":
        print(json.dumps({"train": train_paths, "val": val_paths}, indent=2))
    elif args.emit == "hydra":
        def hydra_list(paths: list[str]) -> str:
            return "[" + ",".join(paths) + "]"

        # shell: eval "$(python ... --emit hydra)"
        train_hydra = "data.train_files=" + hydra_list(train_paths)
        print(f"TRAIN_FILES_ARG={shlex.quote(train_hydra)}")
        if val_paths:
            val_hydra = "data.val_files=" + hydra_list(val_paths)
            print(f"VAL_FILES_ARG={shlex.quote(val_hydra)}")
    else:
        for p in train_paths:
            print(p)
        for p in val_paths:
            print(f"VAL:{p}")


if __name__ == "__main__":
    main()
