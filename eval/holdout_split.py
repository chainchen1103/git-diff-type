#!/usr/bin/env python3
"""Hold some training projects out, to tune settings on projects the model
has not seen without touching the test sets.

Writes the training set without the given repositories, the public-dataset
commits without their organizations, and the held-out commits, into --out.
Train a model on the first two, then tune on the third (eval/tune_prior.py).

Usage (the six projects used for the shipped model):
    python eval/holdout_split.py --out datasets/holdout \\
        --repos aws_aws-cdk commitizen-tools_commitizen element-plus_element-plus \\
                immich-app_immich pnpm_pnpm starship_starship
    python train_enhanced.py --stream --C 0.1 --model datasets/holdout/model_v2.joblib \\
        --data datasets/holdout/train.jsonl datasets/holdout/external.jsonl
    python eval/tune_prior.py --model datasets/holdout/model_v2.joblib \\
        --data datasets/holdout/validation.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repos", nargs="+", required=True, help="repositories to hold out, as miner.py names them")
    ap.add_argument("--train", default="datasets/train.jsonl")
    ap.add_argument("--external", default="datasets/external.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    held = set(args.repos)
    orgs = {name.split("_", 1)[0].lower() for name in held}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "external": 0, "external_dropped": 0}
    with open(args.train, encoding="utf-8") as src, \
            (out / "train.jsonl").open("w", encoding="utf-8") as train, \
            (out / "validation.jsonl").open("w", encoding="utf-8") as validation:
        for line in src:
            if json.loads(line)["repo"] in held:
                validation.write(line)
                counts["validation"] += 1
            else:
                train.write(line)
                counts["train"] += 1
    with open(args.external, encoding="utf-8") as src, \
            (out / "external.jsonl").open("w", encoding="utf-8") as external:
        for line in src:
            if str(json.loads(line).get("owner", "")).split("_", 1)[0].lower() in orgs:
                counts["external_dropped"] += 1
            else:
                external.write(line)
                counts["external"] += 1
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
