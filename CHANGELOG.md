# Changelog

## Unreleased

Builds with the `t5` feature can draft subjects with a small model.

### Added (experimental)

- **Subject drafts from a model** for any change, in builds with the `t5`
  feature (`cargo install --path gca-rs --features t5`) and with a model
  file set by `gca config draft-model <FILE>`, `--draft-model` or
  `GCA_DRAFT_MODEL`. CodeT5-small, fine-tuned on the training commits, reads
  the chosen type and scope, the files, the headers of a few earlier commits
  and the changed lines; gca runs it on the CPU with candle while the type
  prompt is open and offers its draft when the model is sure enough of it,
  else the rules' draft as before. On recent commits of projects gca never
  saw, 14.5% of the commits get a model draft; 34.3% of those are the
  author's subject exactly and 56.0% save at least half the typing.
  `--dry-run` shows the model's draft and confidence, and `--dry-run --json`
  adds `model_draft` and `subject_draft_source`. The model file (67 MB) is
  not released yet.
- `draft_model/` prepares the data, trains the model on an NVIDIA GPU,
  scores it and documents the results; `gca draft-model convert` (in `t5`
  builds) writes a checkpoint as the file gca loads.

## 0.5.1 (2026-10-04)

gca reads the project's history faster where many files come and go, and
prints the same probabilities on every run. The suggestions are otherwise
those of 0.5.0.

### Changed

- **Faster in projects that add and delete many files.** Reading the last
  500 commits, git compared the contents of every added file with every
  deleted one to find files that were moved and edited; gca now has it pair
  only files moved unchanged. Replaying the newest 150 commits of
  shadcn-ui, a dry run took 0.18 s instead of 0.58 s (median, on two cloud
  cores). A file moved and edited in an earlier commit now counts as
  touched under its old path too; the suggestions for those commits, and
  for the newest 150 of vite and astro, are the same.

### Fixed

- With a subject, the probabilities `--dry-run --json` prints could differ
  in their last digits from one run to the next: the subject's words were
  added up in a random order.

## 0.5.0 (2026-10-01)

The diff model also reads whether a change alters behavior. Every other
setting is as in 0.4, and the first suggestion, the top three and the
average recall per type improve on every test set and in every ranking.

### Changed

- **The diff model reads signs of whether a change alters behavior**: the
  shares of the changed lines that were only moved, only renamed or changed
  in literals, only reformatted, or are comments; the shares of the files
  that are new, deleted, renamed or tests; and the words about performance
  in the added lines. Retrained on the same 437,945 commits. On recent
  commits of projects gca never saw, the first suggestion from the diff
  alone is right 45.5% of the time instead of 43.5%, the right type is in
  the top three 86.5% instead of 85.9%, and the average recall per type is
  44.3% instead of 42.9%; with the history and your own commits, the first
  suggestion is right 63.8% instead of 62.3%, and 70.8% instead of 70.2%
  with the subject.
- `refactor` and `perf` stay the hardest. From the diff alone, `perf`
  commits get `perf` first 4.7% of the time instead of 2.7%, and the right
  type is among the three for 26.3% of `refactor` commits instead of 23.4%
  and 25.9% of `perf` commits instead of 22.4%, but `refactor` still never
  comes first. `build` and `ci` lose a little.

### Model and data

- `train_enhanced.py` computes the nine features with ASCII-only rules, and
  `gca-rs` computes them the same way; `gca-rs/tests/behavior_fixtures.json`
  checks the two against each other on tricky diffs. A model file without
  them still loads.
- The prior correction stays at 0.9, the subject's weight and prior power
  and the history's weights as they were. Tuned one at a time for the first
  suggestion, as for 0.4, alpha would have dropped to 0.75: the first
  suggestion from the diff alone gained more, but `perf`, `refactor`,
  `style` and `build` were put first less often, even with the subject and
  the history. On the six held-out training projects, once the history is
  in the ranking, 0.75 gains 0.15 to 0.2 points of first suggestions over
  0.9 and loses 1.0 to 2.5 points of average recall per type.
  `eval/README.md` has both.
- `eval/results_previous.json` now holds gca 0.4's scores.

## 0.4.0 (2026-09-30)

gca learns more from the repository's history: the types the files you are
committing got, and how your own recent commits differ from what it would
have suggested, count toward the type, and the scope prompt follows the
files and your own commits.

### Added

- **Your own commits count toward the type.** gca takes your latest 10
  commits among the last 500, works out what it would have suggested for
  each, and compares that with the type you gave it: a type you choose more
  often than it suggests rises. It reads them from the history every time,
  so nothing is stored. On recent commits of projects gca never saw, with
  each commit's author standing for you, the first suggestion is right 62.3%
  of the time instead of 55.7%, and 70.2% instead of 67.8% with the subject.
  `--dry-run --json` reports how many it read as `ranked_with_own_commits`.
  `eval/tune_history.py` chose the settings and `eval/evaluate_history.py`
  scores them.
- **The commits to the same files count toward the type.** Of the last 500
  commits, gca takes the ones that touched a file you are committing and
  tilts the ranking toward their types too, after the project's own mix. On
  recent commits of projects gca never saw, the first suggestion is right
  55.7% of the time instead of 51.1% with the project's mix alone, and 67.8%
  instead of 66.9% with the subject. `--dry-run --json` reports how many
  commits touched those files as `ranked_with_file_history`.
  `eval/tune_history.py` chose the weights and `eval/evaluate_history.py`
  scores them.

### Changed

- **Better scope suggestions.** A commit without a scope now counts as a
  vote for none, so the scope prompt stays empty where these files usually
  get no scope; commits sharing the deepest directory with the change count
  next after those to the same files; and your own commits count eight
  times. On recent commits of projects gca never saw, the prompt is
  pre-filled exactly right 52.7% of the time instead of 42.1%.
  `eval/evaluate_scope.py` scores it.

### Fixed

- With `log.showSignature` set, git's signature checks were read as files of
  the previous commit in the history gca reads.

## 0.3.0 (2026-09-30)

The type is ranked with more than the diff: the subject when it is known and
the types the project uses. gca also works through plain `git commit`,
follows commitlint configs and drafts the subject of mechanical changes.

### Added

- **commitlint configs.** gca follows a project's `type-enum` and
  `header-max-length` rules: it suggests only allowed types, offers the
  project's own types under "other type…" and with `-t`, and uses the
  project's header limit. JSON configs and `package.json` are read as JSON;
  JavaScript, TypeScript and YAML configs when the rules are written out
  literally. `--dry-run --json` reports them under `commitlint`.
- **The project's history counts toward the type.** gca counts the types
  people gave the last 500 commits and tilts the ranking toward the project's
  own mix. On recent commits of projects gca never saw, the first suggestion
  is right 51.1% of the time instead of 43.5% from the diff alone, and 66.9%
  instead of 61.3% with the subject. `--dry-run --json` reports how many
  commits it read as `ranked_with_history`. `eval/tune_history.py` chose the
  weights and `eval/evaluate_history.py` scores them.
- **`gca config order subject-first`** asks for the subject before the type,
  so the types are ranked with it. `type-first` stays the default. If the
  chosen type and scope then make the header too long, gca asks to shorten
  the subject.
- **`gca hook install`** adds a `prepare-commit-msg` hook, so plain
  `git commit`, editors and git GUIs get the suggested type too: a subject
  from `git commit -m` or a GUI's message box gets the type `gca -y` would
  pick, and the editor opens with the type, any subject draft and the
  ranking. Typed prefixes, merges, reverts, cherry-picks, rebases and amended
  commits keep their messages; the hook never stops a commit, and
  `GCA_HOOK=0` skips it once. `gca hook uninstall` removes it.
- **The subject counts toward the type.** When the subject is known before
  the type, passed with `-m` or drafted, a second model reads it and its
  probabilities are combined with the diff model's. On recent commits of
  projects gca never saw, the first suggestion is right 61.3% of the time
  instead of 43.5%, and the right type is in the top three 90.4% instead of
  85.9%. `--dry-run --json` names the subject it used as
  `ranked_with_subject`. `train_subject.py` trains the model,
  `eval/tune_fusion.py` chose how much it counts, and
  `eval/evaluate_subject.py` scores it.
- **Subject drafts** for mechanical changes: dependency bumps, additions and
  removals in `package.json`, `Cargo.toml`, `pyproject.toml`,
  `requirements*.txt`, `go.mod` and GitHub Actions workflows; lockfile
  updates; releases; renames and moves; deleted files; new or removed tests,
  docs and workflows; one-word typo fixes. The draft is the subject prompt's
  default: Enter takes it, typing replaces it, and Tab puts it on the line to
  edit. A release also pre-selects `chore`. Other changes get no draft.
- `--dry-run` prints the draft; `--dry-run --json` adds `subject_draft`, and
  `from` for a renamed file.

### Fixed

- `-a` and paths could miss a change made right after the last `git add`,
  or report that nothing had changed: the copy of the index they preview in
  looked newer than the index, so git trusted file times it should have
  checked. The copy now keeps the index's time.
- With `diff.renames` turned off, a moved file was read as a deletion plus an
  addition, and its whole content went to the model as changed lines. gca and
  the miner now pin rename detection like the other diff settings.
- With `NO_COLOR` set, the subject prompt showed two colons (`fix::`).
- A mistyped command such as `gca hooks install` showed git's pathspec
  error; gca now says no file matches `hooks` and lists its commands.

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
