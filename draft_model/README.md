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
and of the end token. The drafts it is sure of are much better
(`confidence.py`):

| Confidence | Commits with a draft | Exact | Within one word | Saves half or more |
| --- | ---: | ---: | ---: | ---: |
| -0.2 or more | 9.7% | 47.9% | 56.1% | 67.2% |
| **-0.3 or more** | **14.5%** | **34.3%** | **42.8%** | **56.0%** |
| -0.5 or more | 26.1% | 20.2% | 27.4% | 41.3% |
| any | 100% | 5.5% | 8.1% | 15.9% |

gca offers the model's draft from -0.3 up. This threshold was read off the
same test commits, and projects differ: at -0.3, 6% to 50% of a project's
commits get a draft, and 20% to 78% of those drafts save half the typing or
more. Of the offered drafts that save half the typing, 40% are medusa's
(whose diffs often hold the subject), 16% are `bump version`, `publish` or
a history header repeated, 18% edit a history header, and 26% are new.

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
- A draft takes 0.47 s (median, a full 512-token input, about 8 tokens
  written) on two cores of a 2.1 GHz cloud Xeon, plus 0.3 s to load the
  model; 0.69 s in a two-core virtual machine on a Ryzen AI 9 HX 370 laptop.
  A dry run with the model peaks at 276 MB of memory. gca drafts in the
  background while the type prompt is open, for the type and scope it
  suggests, and drafts again if you pick others.

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
| `baseline_files.py` | the file-overlap baseline |
| `gen_fixtures.py` | the Rust port's test cases, `gca-rs/tests/t5_input_fixtures.json` |
