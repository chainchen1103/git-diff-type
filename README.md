# gca

[中文](./README_CH.md)

Git commit analyzer. gca reads your changes, suggests a Conventional Commit
type, and asks for a subject. By default, it then commits and pushes.
The classifier runs locally with an embedded model.

Supported types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, and `revert`.

## Install

On Windows, run `gca-installer.exe`. It copies `gca.exe` to
`%LOCALAPPDATA%\gca` and adds that directory to your user PATH. If Git is
missing, it attempts to install it with `winget`. Open a new terminal after
installation.

To build from source, install Git and the Rust toolchain, then run:

```sh
cd gca-rs
cargo build --release --bin gca
```

The executable is written to `gca-rs/target/release/gca.exe` on Windows
or `gca-rs/target/release/gca` on other platforms. Add its directory to PATH
or run it by its full path. Python is only needed for training and model
verification.

To build the Windows installer, build `gca` first, then run this from
`gca-rs`:

```sh
cargo build --release --features installer --bin gca-installer
```

## Usage

Run gca inside a Git repository. Configure your Git author identity and
remote as you would for a regular commit and push.

| Command | Behavior |
| --- | --- |
| `gca` | Use staged changes, or stage all changes if nothing is staged; choose a type, enter a subject, commit, and push. |
| `gca ./src tests/foo.py` | Stage and commit only the specified paths, then push. Other staged changes stay staged. |
| `gca list ./src` | Preview which files would be staged without changing the index. |
| `gca --dry-run` | Show suggestions without prompting, changing the index, committing, or pushing. |
| `gca --no-push` | Commit without pushing. |
| `gca --confirm-push` | Ask before pushing. |
| `gca --remote origin` | Push to the specified remote for this run. |
| `gca --topk 5` | Show five suggestions instead of the default three. |
| `gca --model other.json` | Load an exported JSON model instead of the embedded model. |

Place options before paths, for example `gca --no-push ./src`.
`gca list` previews all paths when none are supplied. `gca --dry-run`
uses the same path selection as a normal run, with staging isolated in a
temporary index.

When all selected files match the path rules for `docs`, `test`, or `ci`,
gca preselects that type. You can choose another suggestion.

### Persistent settings

`gca config` writes to your global Git config. Repository settings and
command-line options can override these defaults.

| Command | Behavior |
| --- | --- |
| `gca config push ask` | Ask before each push. |
| `gca config push never` | Commit without pushing. |
| `gca config push auto` | Push without asking; this is the default. |
| `gca config push` | Show the current push setting. |
| `gca config remote upstream` | Use `upstream` as the default remote. |
| `gca config remote` | Show the current remote setting. |

Without a configured remote, gca runs `git push` and uses Git's defaults.
`--no-push` and `--confirm-push` cannot be used together. An invalid
`gca.push` setting stops the commit flow before staging changes.

## Model

The classifier is a calibrated LinearSVC trained on Conventional Commits.
It uses diff text, paths, file extensions, similarity between added and
deleted tokens, and change statistics. Text input is limited to the first
20,000 characters of the diff.

The exported weights in `out/model_v2.json` are embedded at build time.
The installed CLI needs Git, but no Python environment or separate model
file. After exporting new weights, rebuild the CLI or pass the new JSON
file with `--model`.

### Retraining

Use Python 3.11 or newer and run these commands from the project root.
Install the pinned dependencies in your preferred virtual environment:

```sh
python -m pip install -r requirements.txt
```

Collect training data from a local repository:

```sh
python miner.py --repo /path/to/repository --out datasets/local.jsonl
```

To add external data, install `datasets` and import one of the supported
sources. This step requires network access.

```sh
python -m pip install -r requirements-import.txt
python import_external.py --source commitbench --out datasets/commitbench.jsonl
```

Merge the data, train, export, and verify the model:

```sh
python dedupe.py --input datasets --output datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py
python verify_export.py
python gca-rs/gen_fixtures.py --synthetic --out gca-rs/tests/synthetic_fixtures.json
python gca-rs/gen_fixtures.py
cd gca-rs
cargo test --release
cargo build --release --bin gca
```

The miner appends to its output. Deduplication keeps the first copy of each
commit or normalized diff and excludes its own output file. To prioritize
one source, pass its file first with `--input`.

The committed synthetic fixtures let `cargo test` run without a dataset.
After retraining, regenerate both fixture files so their expected
probabilities match the new model.

`verify_export.py` first checks that the JSON parameters match the saved
joblib model, then compares predictions. If the parameters differ, export
the saved model again before verification.

### Evaluation

Training reports metrics on a stratified test split of about 10%, enlarged
for small datasets, and writes the confusion matrix below. The export check
and Rust parity test compare probabilities against the Python model; they
do not measure classification accuracy on new repositories.

![Confusion matrix](out/confusion_matrix.png)

Diffs do not always reveal intent, especially for `refactor`, `fix`, and
`feat`. Review the suggested type before committing.

### Repository and time evaluation

Place local clones of the source repositories under `temp_repos`, each with
an `origin` remote, then run from the project root:

```sh
python evaluate_splits.py --data datasets/_merged.jsonl --repos-dir temp_repos --out out/evaluation
```

The evaluator removes commit headers from diffs, deduplicates by SHA and
normalized diff, and excludes conflicting labels. It compares an 80/20
random split with five folds whose training and test repositories are
disjoint. For commits with dates recovered by SHA from local Git history,
it also compares random and chronological splits of roughly 80/20 on the
same cohort. It uses Git committer timestamps, not the old `labeled_at`
field.

Each split trains a fresh model and leaves `out/model_v2.joblib` and the
exported model unchanged. See the generated [evaluation report](out/evaluation/report.md)
for results and limitations. The evaluation cleanup applies to this script;
it does not change the training command above.

## Development checks

With the Python dependencies installed, run these checks from the project
root:

```sh
python -m unittest discover -s tests -v
cd gca-rs
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```
