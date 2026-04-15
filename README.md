# git-diff-type

[中文](./README_CH.md)

A CLI that classifies your staged git diff into a Conventional Commit type
(`feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`) and prints
a ready-to-copy commit message.

```
$ git add .
$ gca
Stats: +42 / -7 lines in 3 files

? Pick commit type
> feat      ( 71.3%)
  refactor  ( 14.9%)
  chore     (  6.2%)

git commit -m "feat: <description>"
```

## Install

Download `gca.exe` from the latest release, or build from source:

```
cd gca-rs
cargo build --release
# binary: gca-rs/target/release/gca (or gca.exe on Windows)
```

The model weights are embedded in the binary via `include_bytes!`, so the
single executable has no runtime dependencies.

## Usage

```
gca                     # interactive top-3 picker on staged changes
gca --dry-run           # print the top suggestion only
gca --unstaged          # look at unstaged changes
gca --model other.json  # override the embedded model
```

A path-based heuristic runs first (docs / test / ci); mixed commits fall
through to the ML model.

## Retraining

```
python miner.py --repo <path> --out datasets/<name>.jsonl
python import_external.py --dataset commitbench --out datasets/commitbench.jsonl
python dedupe.py --input datasets/*.jsonl --out datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py --model out/model_v2.joblib --out out/model_v2.json
cd gca-rs && cargo build --release
```

`export_model.py` serializes the sklearn pipeline (TF-IDF vocab + IDF,
CountVectorizer vocabs, StandardScaler, LinearSVC coef/intercept per CV
fold, sigmoid calibrators) to JSON. `verify_export.py` re-implements the
forward pass in pure NumPy and asserts agreement with `predict_proba`
within 1e-6.

## Model performance

![confusion_matrix](out/confusion_matrix.png)

`refactor` is the weakest class — the signal mostly lives in intent, not
in the diff surface.
