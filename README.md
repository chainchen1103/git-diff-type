# gca

[中文](./README_CH.md)

**The hardest part of a commit message is often the first word: is this a `feat`, a `fix`, or a `refactor`?**

gca reads the change you are about to commit, ranks the likely Conventional
Commit types with a built-in ML model, and suggests a scope from the project's
own history. You confirm, type the subject line, and gca hands it to
`git commit`.

- **Offline, no API key.** The model is compiled into a single ~11 MB
  executable; your diff never leaves your machine.
- **Measured on projects it never saw.** On 10,005 commits from 18
  repositories that were never used in training, all made after its training
  data ends, the right type is among the three it lists **85.9%** of the
  time. On the 8 of them that the previous version had not seen
  either, its first suggestion is right 45.9% of the time, against
  33.6%.
- **Behaves like git.** Nothing is staged or pushed behind your back; hooks,
  sign-off and your editor work as usual.

A real commit from [shadcn/ui](https://github.com/shadcn-ui/ui), a project the model never saw, replayed with gca:

```
$ git add -A
$ gca
3 files  +188 -1
  A .changeset/fix-registry-header-redirect-leak.md
  A packages/shadcn/src/registry/proxy.test.ts
  M packages/shadcn/src/registry/proxy.ts
✔ Commit type · fix       ( 21.5%)
✔ Scope (optional) · shadcn
✔ fix(shadcn): · drop custom registry headers on cross-origin redirects
[fix-redirect-headers fc3ada1d] fix(shadcn): drop custom registry headers on cross-origin redirects
 3 files changed, 188 insertions(+), 1 deletion(-)
 create mode 100644 .changeset/fix-registry-header-redirect-leak.md
 create mode 100644 packages/shadcn/src/registry/proxy.test.ts
```

Supported types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore` and `revert`.

## Install

**macOS / Linux**:

```
curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh
```

**Windows** (PowerShell):

```
irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex
```

In Command Prompt (cmd), run it through PowerShell:
`powershell -c "irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex"`

The script downloads gca for your platform from the latest
[release](https://github.com/chainchen1103/git-diff-type/releases), checks it
against the release's `SHA256SUMS`, puts it in `~/.local/bin` (Windows:
`%LOCALAPPDATA%\gca`) and adds that folder to your PATH. Nothing else on your
machine changes. Run it again to upgrade. To remove gca, run
`curl -fsSL .../install.sh | sh -s -- --uninstall`, or in PowerShell set
`$env:GCA_UNINSTALL = 1` before the same command. `GCA_VERSION=v0.2.0`
installs a given release, `GCA_INSTALL_DIR` another folder, and
`GCA_NO_MODIFY_PATH=1` leaves your PATH alone.

Prebuilt binaries cover Windows x64 (which also runs on Windows on Arm),
macOS on Apple silicon and Intel, and Linux x86_64. gca needs git.

Other ways to install:

- **Windows installer**: `gca-installer.exe` from Releases does the same as
  the PowerShell script, and offers to install git with `winget` if it is
  missing. `gca-installer --uninstall` removes it.
- **With Rust 1.80 or later**, on any platform:
  `cargo install --git https://github.com/chainchen1103/git-diff-type gca-rs --bin gca`
- **From a clone**: `cd gca-rs && cargo build --release --bin gca` writes
  `target/release/gca` (`gca.exe` on Windows). The installer embeds that
  file, so build it second:
  `cargo build --release --bin gca-installer --features installer`.

Shell completion: `gca completions bash|zsh|fish|powershell`, for example
`gca completions zsh > ~/.zfunc/_gca`.

## Usage

gca works like `git commit`: it commits **what is staged**.

```
git add -p && gca          # pick what belongs in the commit, then run gca
gca -a                     # commit every change to tracked files, like git commit -a (no new files)
gca src/auth tests/auth    # commit only these paths (new files included); other staged work stays staged
```

If nothing is staged, gca does not stage anything for you; it shows
`git status` and how to stage, then exits.

The prompts:

1. **Type**, ranked by probability. The ranking reads the diff, the subject
   when it is already known (from `-m` or a [draft](#subject-drafts)), and
   leans toward the types the project itself used most in its last 500
   commits. Enter accepts the pre-selected one; "other type…" lists the rest.
2. **Scope**, only if the project uses scopes. It is pre-filled with **the
   scope used most often for these files**, learned from the last 500 commits
   (bot commits ignored). Edit or clear it.
3. **Subject**, one line. For a mechanical change gca drafts one, such as
   `bump zod from 3.22.0 to 3.23.8` or `release v1.1.0` (see
   [Subject drafts](#subject-drafts)): Enter takes it, typing replaces it, and
   Tab puts it on the line to edit. A header longer than 100 characters
   (commitlint's default limit) is rejected.

With `gca config order subject-first`, the subject comes first and the types
are then ranked with it, which makes the first suggestion right far more
often (see [With the subject](#with-the-subject)). If the chosen type and
scope make the header too long, gca asks you to shorten the subject.

Esc or Ctrl-C cancels at any point and leaves your staged changes alone.
`-a` and paths are tried out in a temporary copy of the index, so cancelling
leaves the real index exactly as it was.

### Options

```
-t, --type <TYPE>     use this type instead of asking
    --scope <SCOPE>   use this scope instead of asking; "" for none
-m, --message <MSG>   the subject; repeat it for body paragraphs, like git commit -m
-b, --breaking        mark a breaking change (feat!: ...)
-y, --yes             accept the suggested type and scope
-e, --edit            open the message in your editor before committing
-n, --no-verify       skip the pre-commit and commit-msg hooks
-s, --signoff         add a Signed-off-by trailer
    --push            push after committing; sets the upstream if there is none
    --confirm-push    ask before pushing; --no-push does not push
    --remote <NAME>   push to this remote instead of the branch's upstream
    --dry-run         show the suggestions; stage and commit nothing
    --dry-run --json  the same as JSON, for scripts and editor integrations
    --topk <N>        how many suggestions to list (default 3)
    --model <FILE>    use an exported JSON model instead of the built-in one
```

Without prompts, for scripts: `gca -a -t fix -m "handle empty diff"`, adding
`--scope <scope>` (or `--scope ""` for none) in a project that uses scopes, or
`gca -y -m "update guide"` to take the suggested type and scope. Without a
terminal, gca names the flags it still needs instead of hanging.

A subject given with `-m` counts toward the suggested type, which makes the
first suggestion right far more often; see
[With the subject](#with-the-subject).

Exit codes: `0` committed (or dry run done), `1` nothing to commit, cancelled
or git failed, `2` bad arguments, `130` Ctrl-C.

### Settings

Settings live in git config, globally by default, or for the current
repository with `--local`. Options on the command line win.

```
gca config push                 # show the setting (default never: do not push)
gca config push ask             # ask after each commit
gca config push auto --local    # always push in this repository
gca config remote upstream      # push there (default: the branch's upstream, then origin)
gca config order subject-first  # ask for the subject first; it then ranks the types
gca config order type-first     # ask for the type first (default)
```

An invalid `gca.push` or `gca.order` value stops gca before anything is
staged.

### Commit hook

`gca hook install` adds a `prepare-commit-msg` hook to the repository (in
`core.hooksPath`, if that is set), so commits made without gca get a type too:

- `git commit -m "make parsing faster"`, and git GUIs that commit their
  message box as it is, get the type `gca -y` would pick:
  `perf: make parsing faster`. The subject counts toward the type, as with
  `gca -m`.
- `git commit` opens the editor with the type on the first line, followed by
  the subject draft if there is one, and the ranking in a comment below.
- Messages that already start with a type or another `word:` prefix, merges,
  reverts, cherry-picks, rebases and amended commits keep their messages.

The hook never stops a commit. `GCA_HOOK=0 git commit ...` skips it once, and
`gca hook uninstall` removes it. gca does not replace a hook that is already
there; add `gca hook run "$@" || true` to that one instead.

### commitlint

If the project has a commitlint config, gca follows its `type-enum` and
`header-max-length` rules at level 2, the ones that make commitlint reject a
commit. It suggests only the allowed types, lists the project's own types
(such as `deps`) under "other type…" and accepts them with `-t`, and checks
the header against the project's limit instead of 100.

`.commitlintrc`, `.commitlintrc.json` and the `commitlint` key of
`package.json` are read as JSON. JavaScript, TypeScript and YAML configs
cannot be run here, so gca reads the two rules when they are written out
literally, as in `'type-enum': [2, 'always', ['feat', 'fix', 'deps']]`, and
knows that `@commitlint/config-angular` has no `chore`. `--dry-run --json`
shows what it found under `commitlint`.

### Pre-selection rule

When every changed file is documentation, a test, or CI configuration (for
example `docs/`, `*_test.go`, `.github/workflows/`), `docs`, `test` or `ci` is
pre-selected, as long as the model knows that type. You can still pick
another one. A release (see below) pre-selects `chore` the same way.

### Subject drafts

gca drafts the subject only when the staged change follows a pattern:

| Staged change | Draft |
| --- | --- |
| Dependency versions in `package.json`, `Cargo.toml`, `pyproject.toml`, `requirements*.txt`, `go.mod` or a workflow's `uses:` lines | `bump zod from 3.22.0 to 3.23.8`, `downgrade …`, `bump vite and vitest`, `bump 12 dependencies` |
| Dependencies added or removed | `add tempfile dependency`, `remove 3 dependencies` |
| Only lockfiles | `update Cargo.lock`, `update lockfiles` |
| The package's own version, with lockfiles and the changelog | `release v1.1.0`, and `chore` is pre-selected |
| Files moved or renamed without edits | `rename lib.rs to core.rs`, `move util.rs to src/core/`, `rename lib/a/ to lib/b/` |
| Files deleted | `remove scripts/old.sh`, `remove 4 files from legacy/` |
| Tests, a doc or a workflow file added or removed | `add tests for parser`, `add install docs`, `add release workflow` |
| One-word typo fixes in docs | `fix typo in README` |

Anything else gets no draft, including a dependency bump that comes with a
code change: people rarely keep a generic subject such as "update README".
`--dry-run` prints the draft, and `--dry-run --json` has it as
`subject_draft`.

On the three test sets under [Accuracy](#accuracy), 5.2% of the commits people
wrote get a draft (7,402 of 141,213): 3,359 dependency changes, 2,146 releases
and 1,897 moved, removed or added files and typo fixes. For the releases,
authors chose `chore` 96.6% of the time. A draft says what changed, while
authors often wrote why: 6.8% of their subjects match the draft word for word,
so a draft is only a default you can type over.

## Accuracy

The model is trained on **437,945 commits** from 5,925 repositories: 342,347 written by people in 36 open-source projects that follow Conventional Commits, landed before 2026-04-20 (bot commits left out), and 95,598 from the public CommitChronicle and CommitBench datasets, without any organization that owns a test project. Testing uses three separate sets of commits mined from public
repositories:

| Test set | Commits | First suggestion right | Right type in top 3 | Average recall per type | Baseline: always the most common type |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Projects never seen**, commits made after training | 10,005 | **43.5%** | **85.9%** | 42.9% | 44.1% (`fix`) |
| Projects never seen, older history | 91,617 | 44.6% | 83.7% | 36.2% | 35.2% (`fix`) |
| Projects seen in training, commits made after training | 39,591 | 58.0% | 89.7% | 44.9% | 39.2% (`fix`) |

The 18 unseen projects, such as vite, vue, electron, nest, superset and
rolldown, belong to other organizations than any training data, and every
commit whose diff also appears in the training data was removed. All numbers
count commits written by people; bot commits are reported separately in
`eval/results.json`.

*Average recall per type* is the share of each type's commits whose first
suggestion is right, averaged over the types. Always answering `fix` would
make the first suggestion right 44.1% of the time on the first row,
but its average recall per type would be 11.1%, since it never
gets another type right. gca spreads its first suggestions over the types
instead, and relies on the three-item list for the rest.

![Accuracy per type on unseen projects](docs/heldout_accuracy.png)

Compared with the previous model on the same commits (previous → this):

| Test set | First suggestion right | Right type in top 3 | Average recall per type |
| --- | ---: | ---: | ---: |
| Projects never seen, after training | 42.0% → 43.5% | 75.1% → 85.9% | 44.1% → 42.9% |
| … only the 8 projects the previous model never saw either | 33.6% → 45.9% | 68.1% → 86.8% | 42.6% → 47.4% |
| Projects never seen, older history | 41.6% → 44.6% | 73.3% → 83.7% | 41.7% → 36.2% |
| … only the 8 projects the previous model never saw either | 32.2% → 46.4% | 65.7% → 84.8% | 37.4% → 34.6% |
| Projects seen in training, after training | 33.5% → 58.0% | 65.2% → 89.7% | 37.5% → 44.9% |

The previous model's training data included commits from 10 of the 18 test projects (among them quasar, electron, vue core and nest), so for it they were not unseen, and on their older history it was partly scored on commits it had trained on. The 8 projects that neither model saw (hoppscotch, napi-rs, novu, rolldown, shadcn/ui, vitest, vueuse, zitadel) are the fair comparison: there the new model's first suggestion and top three are far better. Average recall per type is where the previous model holds up best, on older history in particular: it put the rarer types first more often, at the cost of being wrong more often overall.

`docs`, `test` and `ci` are the most accurate because the file paths already
carry the signal. `refactor` and `perf` are the weakest: they differ from other
changes in *intent*, which the diff rarely shows. That is why gca offers a
ranked list instead of deciding for you; review the suggestion before you
commit.

### With the subject

When the subject is known before the type, from `-m` or a
[draft](#subject-drafts), gca reads it too: words such as "speed up" or
"rename" state the intent the diff hides. On the same commits, with the subject
each author wrote (without its type prefix):

| Test set | Diff only | Subject only | Both, as gca combines them |
| --- | ---: | ---: | ---: |
| **Projects never seen**, after training | 43.5% · 85.9% · 42.9% | 58.0% · 86.6% · 36.5% | **61.3% · 90.4% · 50.8%** |
| Projects never seen, older history | 44.6% · 83.7% · 36.2% | 55.1% · 85.7% · 34.8% | 57.9% · 88.4% · 45.5% |
| Projects seen in training, after training | 58.0% · 89.7% · 44.9% | 60.5% · 88.2% · 38.7% | 67.9% · 93.6% · 52.0% |

Each cell is first suggestion right · right type in top 3 · average recall per
type. On the first row, `refactor` commits get their own type first 26.6% of
the time instead of never, and `perf` 18.4% instead of 2.7%. The types the
file paths already reveal lose a little (`docs` 77.7% → 75.7%, `ci`
77.9% → 76.7%), `test` more (71.2% → 58.7%); the path rule still pre-selects
those three when every file fits.

The subject model is logistic regression over words and pairs of words,
trained on the subjects of the same commits as the diff model
(`train_subject.py`). gca multiplies the two models' probabilities, with the
subject's raised to the power 0.25 and each type's training frequency divided
out to the power 0.15. Both settings were chosen on six training projects held
out for the purpose, not on the test sets. To have the subject count on every
commit, ask for it first: `gca config order subject-first`.

### With the project's history

Projects have habits: some file dependency updates under `chore`, others
under `build`; some never write `refactor`. gca counts the types people gave
the last 500 commits (bots left out) and tilts the ranking toward that mix,
as far as it differs from the mix gca was trained on. Files have habits too,
so it then tilts the ranking the same way toward the types of the commits
among those 500 that touched a file you are committing. Scored so that each
test commit sees only the commits before it:

| Test set | Diff only | Diff and history | Diff and subject | Diff, subject and history |
| --- | ---: | ---: | ---: | ---: |
| **Projects never seen**, after training | 43.5% · 85.9% · 42.9% | 55.7% · 91.4% · 53.0% | 61.3% · 90.4% · 50.8% | **67.8% · 94.2% · 58.5%** |
| Projects never seen, older history | 44.6% · 83.7% · 36.2% | 54.6% · 90.0% · 42.4% | 57.9% · 88.4% · 45.5% | 64.6% · 93.4% · 50.3% |
| Projects seen in training, after training | 58.0% · 89.7% · 44.9% | 59.9% · 92.6% · 50.2% | 67.9% · 93.6% · 52.0% | 68.5% · 94.8% · 55.1% |

(first suggestion right · right type in top 3 · average recall per type).
On the projects never seen, the project's mix alone gets the first suggestion
right 51.1% of the time from the diff and 66.9% with the subject; the commits
to the same files add the rest. History helps most on projects the model
never saw; on the ones it trained on, it had already learned their habits.
How much each part counts was chosen on the six held-out training projects:
the project's mix has a weight of 0.1, or 0.25 with a subject, smoothed with
10 commits' worth of the training mix; the same files' mix 0.1, or 0.15 with
a subject, smoothed with 5. A project that uses types the way the training
data did keeps its ranking, and one without Conventional Commits in its
history is not affected.

Full numbers and how to reproduce them: [eval/README.md](eval/README.md).

## How it works

![Architecture](docs/architecture.png)

1. **Data.** `miner.py` extracts (diff, type) pairs from repositories that
   follow Conventional Commits; `split_dataset.py` separates training data and
   the three test sets by project and date. `import_external.py` and
   `eval/prepare_external.py` add public datasets without touching the test
   projects.
2. **Training.** TF-IDF over the diff, bag-of-words over file paths and
   extensions, the overlap between added and removed words, and line counts
   feed a calibrated LinearSVC that outputs a probability for each of the 11
   types. Only the first 20,000 characters of a diff are read. Each
   probability is then corrected for how common the type was in training
   (`eval/tune_prior.py`); otherwise most diffs would be called `fix`, and
   `refactor` or `perf` would almost never come first. A second, smaller
   model reads the subject when it is known before the type:
   logistic regression over its words and pairs of words
   (`train_subject.py`, see [With the subject](#with-the-subject)).
3. **Export.** `export_model.py` flattens the whole sklearn pipeline into
   `out/model_v2.json`; `train_subject.py` writes `out/subject_model.json`.
4. **Inference.** The Rust CLI embeds both JSON files at build time and
   reimplements the forward passes, so the installed CLI needs git but no
   Python or model file. `gca-rs/tests/parity.rs` checks that Rust matches
   Python within 1e-6 (1e-9 for the subject model). At run time gca also
   reads the repository's last 500 commits, for the scope and for the
   types the project and the files you are committing usually get.
   After exporting new weights, rebuild the CLI or pass the JSON with `--model`.

gca always reads the diff in git's default format, whatever personal settings
such as `diff.noprefix` or `color.ui` say, so it matches the training data.

### Retraining

Needs Python 3.11+ and `pip install -r requirements.txt`. From the project root:

```
bash eval/collect.sh                    # clone eval/repos_*.txt, mine, split into datasets/
pip install -r requirements-import.txt  # public datasets (network access)
python import_external.py --source commitchronicle --out datasets/commitchronicle.jsonl
python import_external.py --source commitbench --out datasets/commitbench.jsonl
python eval/prepare_external.py --inputs datasets/commitchronicle.jsonl datasets/commitbench.jsonl \
    --test-repos eval/repos_test.txt --out datasets/external.jsonl \
    --against datasets/train.jsonl datasets/test_*.jsonl
python eval/tune_c.py --data datasets/train.jsonl   # choose C on training projects only
python train_enhanced.py --stream --data datasets/train.jsonl datasets/external.jsonl --C 0.1
python eval/tune_prior.py --write 0.9   # correct for how common each type is (eval/README.md)
python export_model.py
python verify_export.py --data datasets/test_unseen_recent.jsonl
python gca-rs/gen_fixtures.py --synthetic --out gca-rs/tests/synthetic_fixtures.json
python gca-rs/gen_fixtures.py --data datasets/test_unseen_recent.jsonl
bash eval/run.sh                        # score the three test sets
python eval/tune_fusion.py ...          # how much the subject counts (eval/README.md)
python train_subject.py --data datasets/train.jsonl datasets/external.jsonl --weight 0.25 --prior-power 0.15
python train_subject.py --write-fixtures gca-rs/tests/subject_fixtures.json \
    --fixture-data datasets/test_unseen_recent.jsonl
python eval/evaluate_subject.py --sets datasets/test_*.jsonl
python eval/tune_history.py ...         # how much the project's history counts (eval/README.md)
python eval/evaluate_history.py --sets datasets/test_*.jsonl --history datasets/train.jsonl
cd gca-rs && cargo test --release && cargo build --release --bin gca
```

`verify_export.py` first checks that the JSON parameters match the saved
joblib model, then compares predictions; `--check-fixtures` compares the JSON
with the parity fixtures instead. The committed synthetic fixtures let
`cargo test` run without a dataset; `gca-rs/tests/fixtures.json`, built from
real diffs, stays local. Regenerate both after retraining.

To train on your own history instead, mine a repository with
`python miner.py --repo <path> --out datasets/local.jsonl` (it appends),
merge sources with `python dedupe.py --input datasets --output datasets/_merged.jsonl`
(the first copy of each commit or normalized diff wins, so pass the preferred
source first), and train without `--stream` on the merged file; that also
reports a stratified 10% split and writes `out/confusion_matrix.png`.

### Other evaluations

`evaluate_splits.py` trains fresh models to compare a random 80/20 split with
five folds whose training and test repositories are disjoint and, for commits
whose dates can be read from local clones, a random split with a chronological
one:

```
python evaluate_splits.py --data datasets/_merged.jsonl --repos-dir temp_repos --out out/evaluation
```

It removes commit headers from diffs, deduplicates by SHA and normalized diff,
drops conflicting labels, and leaves the shipped model unchanged.

## Development checks

```
python -m unittest discover -s tests -v
cd gca-rs
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

CI runs them on every push, the Rust checks on Windows, macOS and Linux, and
builds the installer on Windows.

## Roadmap

- Draft the subject of other commits too, still offline: a small built-in
  model that completes the subject after the chosen type and scope (mechanical
  commits already get [drafts](#subject-drafts))
- Learn from the types a user picks over the suggestion, beyond the types
  the project and its files already get, which gca reads from the history
- Better `refactor` / `perf` with features about whether behavior changed
- Notice staged changes that mix unrelated work and suggest splitting them
- Suggest `!` for breaking changes. Spotting removed exports is not enough:
  on the unseen projects only 0.5% of commits are marked breaking, and of the
  3.7% that remove a public definition, 4.1% are (`eval/breaking_signal.py`)

[Changelog](CHANGELOG.md)
