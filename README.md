# git-diff-type

[中文](./README_CH.md)

A CLI that classifies your staged git diff into a Conventional Commit type
(`feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`) and prints
a ready-to-copy commit message.

```
$ gca
no staged changes; running `git add -A`
Stats: +42 / -7 lines in 3 files

? Commit type
> feat      ( 71.3%)
  refactor  ( 14.9%)
  chore     (  6.2%)

? feat: add user login middleware
[main a1b2c3d] feat: add user login middleware
 3 files changed, 42 insertions(+), 7 deletions(-)
```

One command from dirty tree to pushed commit: auto-stages if nothing
is staged, lets you pick the type, prompts for the subject line,
commits, and pushes.

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
gca                     # default: stage all if empty, pick, commit, push
gca ./src tests/foo.py  # stage only these paths, then commit + push
gca --no-push           # one-off: commit only
gca --confirm-push      # one-off: ask before pushing
gca --dry-run           # print the resulting message without committing
gca --model other.json  # override the embedded model
```

Persistent push behavior lives in `git config` under `gca.push`:

```
gca config push ask     # prompt before push every time
gca config push never   # commit only
gca config push auto    # push without asking (default)
gca config push         # show current value

gca config remote upstream  # always push to upstream
gca config remote           # show current remote
gca --remote origin         # one-off: push to origin this time
```

CLI flags still override the stored setting for a single invocation.
A path-based heuristic (docs / test / ci) sets the default cursor
position when all staged files match one rule; you can still pick
anything.

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
