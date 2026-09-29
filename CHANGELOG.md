# Changelog

## 0.2.0 (2026-09-30)

gca now follows `git commit`'s conventions: it commits what you staged, never
stages or pushes on its own, and respects hooks and git settings. The model is
retrained and measured on commits from projects it never saw.

### Changed

- **Nothing is staged automatically.** With nothing staged, gca shows
  `git status` and how to stage, and exits with 1. `-a` commits every change
  to tracked files, like `git commit -a`; naming paths commits only those,
  like `git commit <path>...`.
- **Nothing is pushed by default.** Pass `--push`, or set
  `gca config push auto` (or `ask`), globally or per repository with
  `--local`. An existing `gca.push` setting keeps working; an invalid one is
  now an error, reported before anything is staged.
- `-a` and paths are previewed in a scratch copy of the index, so cancelling
  leaves what you staged exactly as it was. The preview also works with a
  custom `GIT_INDEX_FILE`, a split index and linked worktrees.
- `--dry-run` prints the ranking without prompting and never stages.
- `gca list` is gone; `gca --dry-run [-a | <path>...]` lists the files.
- Options after paths work (`gca src -m "..."`); they used to be read as paths.
- The path rule (`docs`, `test`, `ci`) only pre-selects a type the model knows.

### Added

- One-line install, like other developer tools:
  `curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh`
  on macOS and Linux, and
  `irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex`
  in PowerShell on Windows. The scripts download the release binary for the
  platform, check it against the release's `SHA256SUMS` and put it on PATH.
  Running one again upgrades gca; `--uninstall` (`$env:GCA_UNINSTALL = 1` in
  PowerShell) removes it and its PATH entry. CI runs both on Linux, macOS and
  Windows, including Windows PowerShell 5.1, and again against each release.
- Scope suggestions learned from the repository's history; gca asks for a
  scope only when the project uses them. `--scope` sets it directly.
- `--breaking` (`!`), body paragraphs with repeated `-m`, `-e/--edit`,
  `-n/--no-verify`, `-s/--signoff`.
- `--type`, `--yes` and `-m` for commits without prompts, and a clear error
  instead of a crash when there is no terminal.
- "other type…" in the menu for types outside the top suggestions.
- `--dry-run --json` for scripts and editor integrations.
- `--push` sets the upstream when the branch has none.
- `gca completions <shell>` and `--version`.
- Windows installer: asks before installing Git, `--uninstall`, `--yes`.
- Builds on macOS and Linux; releases include Windows, macOS (Apple silicon
  and Intel) and static Linux binaries. CI lints and tests on all three.

### Fixed

- Predictions degraded for anyone with `diff.noprefix`, `diff.mnemonicPrefix`
  or `diff.relative` set, because the file paths the model relies on were
  missing or different. gca and the miner now read diffs in git's default
  format regardless of settings.
- The CLI split some text into words differently from the Python model it
  was trained with: Rust's `\w` counts combining marks (such as the dot in a
  lowercased Turkish "İ", or a decomposed accent) as part of a word, Python's
  does not. The tokenizers, the word-overlap feature, token patterns with a
  capture group and l1 normalization now match scikit-learn, with tests.
- A malformed or mismatched model file is rejected with a clear error before
  anything is staged, instead of panicking or predicting garbage. The
  calibration no longer overflows on extreme scores, and two-class models work.
- Line counts no longer overflow on huge diffs.
- Esc and Ctrl-C in a prompt now cancel cleanly and restore the cursor.
- gca refuses to run during a merge, cherry-pick or revert, where git has
  already prepared the message.
- `--dry-run` no longer needs a terminal.
- The installer ignores an empty `LOCALAPPDATA` and refuses to run outside
  Windows.

### Model and data

- Retrained on 437,945 commits from 5,925 repositories: 342,347 written by people in 36 projects mined for this release, and 95,598 from CommitChronicle and CommitBench without the test projects' organizations.
- The previous model's own mined commits had the commit header, and so the answer, inside the diff (60,403 of them); the new data is mined without it.
- Evaluated on three held-out sets totalling 166,417 commits, including
  18 projects never used in training. See `eval/README.md`.
- On the 10,005 commits by people in the 18 unseen projects, the first suggestion is right 43.5% of the time (previous model: 42.0%), the right type is in the top three 85.9% of the time (previous: 75.1%), and recall averaged over the types is 42.9% (previous: 44.1%). On the 8 of those projects that the previous model had not trained on either: 45.9%, 86.8% and 47.4% (previous: 33.6%, 68.1% and 42.6%).
- The data pipeline is reproducible: `eval/collect.sh` clones the projects in
  `eval/repos_*.txt` and mines them with `miner.py --stream`, which reads each
  repository in two `git log` passes instead of starting git twice per commit
  and skips commits at the edge of a shallow clone, whose diffs would show the
  whole repository as added. `split_dataset.py` builds the training and test
  sets, and `eval/prepare_external.py` adds the public datasets.
- `train_enhanced.py --stream` fits the vocabularies on a sample and
  transforms the corpus in chunks, so hundreds of thousands of commits train
  in a few GB of memory. `eval/tune_c.py` chooses the settings on training
  data only.
- Each type's probability is divided by its share of the training data
  raised to the power 0.9, chosen on training projects held out from a
  separate model (`eval/tune_prior.py`, `eval/holdout_split.py`). Without it
  the model put `fix` first for 71.1% of the commits in unseen
  projects and almost never put `refactor`, `perf` or `style` first.
- The committed `model_v2.json`, `model_v2.joblib` and parity fixtures come
  from the same training run. `verify_export.py` checks the JSON against the
  joblib, and CI checks it against the fixtures.
- `evaluate_splits.py` compares random, repository-disjoint and
  chronological splits. `dedupe.py` keys commits by owner, repository and
  SHA, reads JSON arrays and wrapped objects, and its near-duplicate search no
  longer misses pairs whose differing bits are spread out. `import_external.py`
  counts deleted and binary files and code lines that begin with `---` or
  `+++`.
- Python dependencies are pinned, and the pipeline has unit tests.

## 0.1.0

First Rust release: one command from a dirty working tree to a pushed commit,
with the model embedded in the executable.
