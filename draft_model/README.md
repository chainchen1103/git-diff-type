# Subject model and type model

Two small models: one writes the subject (below), which gca can use in
builds with the `t5` feature, the other tells the type ([Type model](#type-model)).

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

## Type model

The same network's encoder, fine-tuned to tell the type: its states,
averaged over the input's tokens, go through one linear layer to the eleven
types.

### Data and training

`prepare_type.py` writes `type_data/` from the same commits, shuffle and
validation split as `prepare.py`: 435,905 to train on, 2,000 held back.
The input is the subject model's without its first two lines, the type and
the scope, which are what the model tells. The history lines are picked
without the type, as gca can pick them before you choose one: up to three
of the 500 earlier commits that touched a staged file (the larger overlap
first, then the newer), the newest commits of any type if fewer than three
did, and the author's own newest commit. Their headers carry their types,
the project's habits as gca reads them. For the three held-out sets it
writes every commit gca's evaluation scores, with its history from the
commits before it.

`run_type.bat` trains (`train_type.py`) and writes the predictions for the
held-out sets (`predict_type.py`). It starts from the subject model's
encoder (`runs/`), which has read these inputs before: one epoch, 13,623
steps of 32 commits, AdamW at 2e-4 with 500 warm-up steps and a linear
decay, bfloat16, about 1.6 hours and 6.1 GB of GPU memory on an RTX 4060
Laptop. The validation accuracy, on commits of the training projects,
rises from 65.8% at step 1,000 to 73.0%.

### Results

`eval_type.py` ranks each held-out commit as gca does, with the built-in
linear model, with the type model, and with both averaged in log space,
half each. First suggestion right · right type in the top three · average
recall per type · average F1 per type (types with 20 commits or more):

| Recent commits of the unseen projects (10,005) | Model alone | With history and own commits | With the subject | Subject, history and own commits |
| --- | ---: | ---: | ---: | ---: |
| Built-in model (0.5) | 45.5% · 86.5% · 44.3% · 34.6% | 63.8% · 93.0% · 56.2% · 53.7% | 61.7% · 90.8% · 51.6% · 48.9% | 70.8% · 94.9% · 60.3% · 61.2% |
| Type model | 67.7% · 94.4% · 60.5% · 60.8% | 69.3% · 95.0% · 61.0% · 62.0% | 69.4% · 95.0% · 61.4% · 61.9% | 72.1% · 95.7% · 62.8% · 64.4% |
| **Both** | 67.7% · 94.4% · 60.6% · 60.7% | **70.4% · 95.3% · 62.0% · 63.2%** | 70.2% · 95.1% · 61.6% · 61.7% | **73.7% · 96.1% · 63.5% · 65.4%** |

| Older commits of the unseen projects (91,617) | Model alone | With history and own commits | With the subject | Subject, history and own commits |
| --- | ---: | ---: | ---: | ---: |
| Built-in model (0.5) | 45.9% · 84.3% · 36.9% · 27.8% | 60.8% · 92.1% · 44.6% · 44.2% | 58.1% · 88.6% · 45.8% · 40.1% | 67.9% · 94.6% · 50.5% · 51.9% |
| Type model | 64.9% · 93.2% · 46.8% · 47.0% | 66.4% · 93.8% · 47.5% · 48.5% | 66.7% · 93.8% · 48.3% · 48.3% | 68.8% · 94.7% · 49.3% · 50.7% |
| **Both** | 64.8% · 93.2% · 46.7% · 46.6% | **67.1% · 94.2% · 47.6% · 48.9%** | 67.2% · 93.9% · 48.8% · 48.6% | **70.2% · 95.3% · 50.1% · 51.9%** |

"Model alone" is not the same input for the two: the type model reads the
headers of related earlier commits too, the built-in model only the diff.

- On the recent commits, with history and own commits, F1 rises for
  `refactor` from 24% to 45%, `test` 58% to 81%, `build` 42% to 61%,
  `chore` 55% to 64% and `fix` 72% to 78%. `perf` does not gain (36% and
  35%; 47% and 38% with the subject), and on the older commits `style` and
  `revert` drop (17% to 11%, 6% to 0%): the type model rarely puts them
  first.
- gca reads your own earlier commits again to see how the types you gave
  them differ from what it would have suggested. With the type model it
  still reads them with the built-in model alone, which saves running the
  type model ten more times: scored that way, the first suggestion is right
  70.4% of the time on the recent commits, against 70.5% with both models,
  and 67.1% on the older ones either way.
- The weights of the history are gca's own, and the type model's
  probabilities are not corrected for how common each type is. Chosen on
  `test_seen_recent` instead (the training projects' recent commits),
  other weights would have gained 0.7 points without the subject and lost
  0.2 with it on the recent commits.

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
| `prepare_type.py` | builds `type_data/` for the type model from `datasets/` |
| `run_type.bat` | trains the type model and writes its predictions (Windows, NVIDIA GPU) |
| `train_type.py`, `predict_type.py` | fine-tunes the type model (`--resume` continues), writes `type_predictions/` |
| `eval_type.py` | the type model's tables above |
