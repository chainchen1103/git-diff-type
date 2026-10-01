# Subject model

A small sequence-to-sequence model that writes the subject of a Conventional
Commit: [CodeT5-small](https://huggingface.co/Salesforce/codet5-small)
(60.5 million parameters) fine-tuned on the same mined commits as the type
model. It reads the type and scope chosen before the subject, the changed
files, the headers of up to four earlier commits in the project, and the
changed lines; it writes the subject. gca built with the `t5` feature runs it
on the CPU and offers its draft when the model is sure enough of it
([Subject drafts](../README.md#drafts-from-a-model-experimental)).

## Data

`prepare.py` builds the data from `datasets/`, the files `eval/collect.sh`
and `import_external.py` produce (see [eval/README.md](../eval/README.md)):

- **Training**: the commits by people in `train.jsonl` and `external.jsonl`
  with a type, a diff and a subject of at most 200 characters: 437,905, of
  which 2,000 are held back for validation, shuffled.
- **Test**: the 10,005 commits by people in `test_unseen_recent.jsonl`, from
  18 projects in other organizations, landed on or after 2026-04-20.
- **Target**: the subject as the author wrote it, without the
  `type(scope):` prefix or a trailing pull request number.
- **Input** (`t5_format.py`): `type:`, `scope:`, the files with their status
  (at most 12 named), the history lines, then each file's hunk headers and
  changed lines (each cut at 160 characters), cut at 2,000 characters. Most
  inputs still fill the model's 512 tokens: 70% are longer and lose their
  end.
- **History**: up to three earlier commits in the project that touched a
  staged file (the same type first, then the larger overlap of files, then
  the newer), the newest of the same type if fewer than three did, and the
  author's own newest commit, from the 500 before it. gca can pick them from
  what it already reads: subjects, files and authors, no diffs. 94.1% of the
  training commits and all test commits have at least one.

`prepare.py` rebuilds the files the published model was trained on byte for
byte; `--no-history` builds the inputs without the history lines.

## Training

`run.bat` sets up a Python environment with PyTorch for the NVIDIA GPU
(`setup_env.py`), trains (`train_t5.py`) and writes the test predictions
(`generate_t5.py`). One epoch, 13,623 steps of 32 commits, AdamW at 3e-4 with
500 warm-up steps and a linear decay, bfloat16: about 2.2 hours and 4.4 GB of
GPU memory on an RTX 4060 Laptop. Checkpoints go to `runs/` every 1,000
steps; Ctrl+C saves one and running `run.bat` again continues from it.

The validation loss ends at 2.789 with the history lines and 2.896 without.

## Results

On the 10,005 test commits, against the subjects the authors wrote, ignoring
case, extra spaces and a final period (`score.py`):

| Suggestion | Exact | Within one word | Saves half the typing or more | First word right |
| --- | ---: | ---: | ---: | ---: |
| Model, without history lines | 2.7% | 5.2% | 12.0% | 22.1% |
| **Model, with history lines** | **5.5%** | **8.1%** | **15.9%** | **26.4%** |
| Model with history lines, best of three (beam search) | 6.8% | 10.6% | 22.0% | 28.2% |
| The earlier commit of the type with the most files in common (`baseline_files.py`) | 1.9% | 3.8% | 5.7% | 10.7% |

"Saves half the typing or more": the character edits that turn the draft
into the author's subject are at most half its length. Given the first word
of the author's subject, the model's next word is right 27.1% of the time
with the history lines, 22.1% without.

The history lines help most where a project repeats itself:

- `chore` subjects are exact 17.4% of the time with them and 2.7% without;
  `feat` 6.8% and 3.7%; `fix`, the most common type, 3.9% and 3.5%.
- For 6% of the commits (602) the author's subject, or one a word away from
  it, is among the history lines. The model writes it 44% of the time, often
  by editing it: from `publish v11.1.23 release` it writes
  `publish v11.1.26 release`, which copying the line cannot. For the other
  94% the history lines add less: exact 3.1% instead of 2.7%, half the typing
  saved 12.8% instead of 11.4%.
- One project dominates the exact matches without history: medusa's diffs
  often hold the subject itself, in a changeset file. Without medusa, 3.3%
  of the subjects are exact with history lines and 0.5% without.
- 111 commits share their commit time (to the second) with an earlier commit
  that was picked for their history lines, which could in fact be a later
  one; without them the scores are the same.

### Confidence

The model's confidence in a draft is the mean log-probability of its tokens
and of the end token. The drafts it is sure of are much better, except one
kind: when a change touches the same files as the commit before, the model
sometimes copies that commit's subject. Of the 1,454 drafts with a
confidence of -0.3 or more, 155 repeat the subject of exactly one of the
500 earlier commits, and only 2.6% of those are exact (23.2% save half the
typing); the 160 that repeat a subject several earlier commits share, such
as `bump version`, are exact 80.0% of the time. gca holds back the first
kind (`confidence.py`):

| Confidence | Commits with a draft | Exact | Within one word | Saves half or more |
| --- | ---: | ---: | ---: | ---: |
| -0.2 or more | 8.9% | 52.1% | 60.1% | 71.0% |
| **-0.3 or more** | **13.0%** | **38.0%** | **46.7%** | **59.9%** |
| -0.5 or more | 23.1% | 22.5% | 29.7% | 43.6% |
| any | 96.1% | 5.6% | 8.1% | 15.7% |

By confidence alone, -0.3 would give 14.5% of the commits a draft, 34.3% of
them exact and 56.0% saving half the typing.

gca offers the model's draft from -0.3 up. This threshold was read off the
same test commits, and projects differ: at -0.3, 4% to 49% of a project's
commits get a draft, and 21% to 78% of those drafts save half the typing or
more. Of the offered drafts that save half the typing, 41% are medusa's
(whose diffs often hold the subject), 17% are `bump version`, `publish` or
a history header repeated, 17% edit a history header, and 25% are new.

### On real repositories

`replay.py` shows what gca offers as a person committing would see it: each
commit is staged on top of its parent, under its author's name, and
`gca --dry-run --json` is asked with the author's type and scope.

| Repository | Commits by people | Model drafts offered | Exact | Saves half or more |
| --- | ---: | ---: | ---: | ---: |
| gca itself, its newest | 60 | 2 | 2 | 2 |
| shadcn-ui/ui, since 2026-04-20 | 150 | 78 | 14 | 46 |
| commitlint, since 2026-04-20 | 29 | 3 | 0 | 1 |

On gca's own commits the two drafts are its releases (`release 0.5.0`,
`release 0.4.0`); without holding back repeats it would also have offered
the subjects of the commits before `feat(model): retrain with the behavior
features` and `feat(cli): add a prepare-commit-msg hook`. Its other drafts,
not offered, mostly say what changed (`update README`) where the author
wrote why. In shadcn-ui most drafts fill in its template, `add @name to the
registry directory`.

## In gca

gca reads the model from one GGUF file with the tokenizer inside: the
matrices in 8-bit blocks (Q8_0), 67 MB. From a checkpoint folder:

```
cd gca-rs
cargo run --release --features t5 -- draft-model convert ../draft_model/runs/ckpt-a gca-draft-q8.gguf
```

- `gca-rs/src/t5draft/input.rs` lays out the input and picks the history
  lines as `t5_format.py` and `prepare.py` do: `gen_fixtures.py` writes the
  cases its test compares, and a one-off run with 3,039 inputs and 3,000
  history cases from the test commits matched them all.
- `gca-rs/src/t5draft/runtime.rs` runs the network with
  [candle](https://github.com/huggingface/candle) in 32-bit floats. With
  the checkpoint's own weights it writes the same subjects as transformers on
  the CPU (60 of 60 test commits, the same token ids, confidence within
  0.0002). With the 8-bit file, on 1,000 test commits, it scores as the GPU
  run did: exact 6.1% both, saving half the typing 16.5% against 16.6%, and
  the same 14.7% of drafts offered at -0.3.
- A draft takes 0.19 s (median of 60 test commits, most with a full
  512-token input, about 8 tokens written) on a Ryzen AI 9 HX 370 laptop
  under Windows, with the same subjects and confidences as on Linux; 0.47 s
  on two cores of a 2.1 GHz cloud Xeon, plus 0.3 s there to load the model,
  and 0.69 s in a two-core virtual machine on the laptop. A dry run with the
  model peaks at 276 MB of memory. gca drafts in the background while the
  type prompt is open, for the type and scope it suggests, and drafts again
  if you pick others.

## Files

| File | What it does |
| --- | --- |
| `prepare.py` | builds `data/` from `datasets/` |
| `t5_format.py` | lays out one input |
| `run.bat` | sets up, trains and writes predictions (Windows, NVIDIA GPU) |
| `setup_env.py`, `t5_common.py` | the environment, and helpers shared by the two scripts below |
| `train_t5.py` | fine-tunes the model; `--resume` continues |
| `generate_t5.py` | writes `predictions.jsonl`: greedy subjects with their confidence, three by beam search, and completions |
| `score.py`, `confidence.py` | the tables above |
| `replay.py` | replays a repository's commits through gca (`python draft_model/replay.py GCA REPO MODEL N OUT.jsonl`); it adds and removes a temporary git worktree |
| `baseline_files.py` | the file-overlap baseline |
| `gen_fixtures.py` | the Rust port's test cases, `gca-rs/tests/t5_input_fixtures.json` |
