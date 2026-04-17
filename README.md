# gca — Git Commit Analyzer

[中文](./README_CH.md)

One command to go from dirty working tree to pushed commit.
Analyzes the staged diff with ML to suggest a Conventional Commit type
(`feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`),
prompts for the subject line, commits, and pushes.

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

## Install

Run `gca-installer.exe` — it copies `gca.exe` to
`%LOCALAPPDATA%\gca`, adds the directory to your user PATH, and checks
that git is installed (offers to install via `winget` if missing).

Or build from source:

```
cd gca-rs
cargo build --release --bin gca
# binary: target/release/gca.exe
```

## Usage

```
gca                        # auto-stage all → pick type → commit → push
gca ./src tests/foo.py     # stage only these paths → commit → push
gca --dry-run              # print the suggestion without committing
gca --no-push              # commit only, skip push
gca --confirm-push         # ask before push
gca --remote origin        # push to a specific remote this time
gca --model other.json     # use a different model file
```

### Persistent settings

Settings are stored via `git config` and persist across invocations.
CLI flags override them for a single run.

```
gca config push ask         # always ask before pushing
gca config push never       # never push (commit only)
gca config push auto        # push without asking (default)
gca config push             # show current value

gca config remote upstream  # always push to upstream
gca config remote           # show current remote
```

### Heuristic

A path-based heuristic covers `docs`, `test`, and `ci` — when all
staged files match one rule, the corresponding type is pre-selected.
You can still pick a different type.

## Model

The classifier is a calibrated LinearSVC trained on Conventional Commit
repositories. Model weights are embedded in the binary at build time,
so the executable is fully self-contained (~11 MB).

### Retraining

```
python miner.py --repo <path> --out datasets/<name>.jsonl
python import_external.py --dataset commitbench --out datasets/commitbench.jsonl
python dedupe.py --input datasets/*.jsonl --out datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py
cd gca-rs && cargo build --release --bin gca
```

### Performance

![confusion_matrix](out/confusion_matrix.png)

`refactor` is the weakest class — the signal mostly lives in intent,
not in the diff surface.
