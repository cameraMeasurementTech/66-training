#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# SPDX-License-Identifier: Apache-2.0
"""Convert JSONL (one object per line) with a `messages` field into Parquet for MultiTurnSFTDataset."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", type=Path, help="Input .jsonl; each line must include a 'messages' list")
    parser.add_argument("output_parquet", type=Path, help="Output .parquet path")
    parser.add_argument(
        "--messages-key",
        default="messages",
        help="Field name for chat turns (default: messages, OpenAI-style list of {role, content})",
    )
    args = parser.parse_args()

    rows = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if args.messages_key not in obj:
                raise KeyError(f"Line {i + 1}: missing key {args.messages_key!r}")
            rows.append({args.messages_key: obj[args.messages_key]})

    df = pd.DataFrame(rows)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {len(df)} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
