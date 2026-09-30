# Evaluation

How well the shipped model (`out/model_v2.json`, the one compiled into `gca`)
predicts the type people chose, on commits it was never trained on.

## Data

### Mined projects

`eval/collect.sh` clones the repositories listed in `eval/repos_train.txt`
and `eval/repos_test.txt` (history since 2018), mines every commit whose
subject follows Conventional Commits with `miner.py --stream`, and splits them
with `split_dataset.py`:

| Set | What it is | Commits | Written by people |
| --- | --- | ---: | ---: |
| `train` | 36 training projects, landed before 2026-04-20 | 342,347 | 342,347 |
| `test_unseen_recent` | 18 other projects, landed on or after 2026-04-20 | 12,990 | 10,005 |
| `test_unseen_older` | the same 18 projects, landed before 2026-04-20 | 108,456 | 91,617 |
| `test_seen_recent` | the training projects, landed on or after 2026-04-20 | 44,971 | 39,591 |

- **Label**: the type the author wrote in the commit subject.
- **Landed** is the committer date, when the commit reached the branch.
- The training and test projects belong to different GitHub organizations.
- A test commit whose normalized diff also appears in the training set is
  dropped (8 were), and each set is deduplicated
  (2,684 duplicates removed).
- Commits by bots (Renovate, Dependabot, GitHub Actions and the like) are
  left out of training (40,668 of them) and scored separately; the headline
  numbers are for commits written by people.

### Public datasets

`import_external.py` imported 74,660 commits from CommitChronicle and 23,126 from CommitBench. `eval/prepare_external.py` removed those from organizations that own a test project (1,329 and 823) and those whose normalized diff was already in the mined sets or earlier in the imports (7 and 29), leaving 95,598 commits from 5,911 repositories for training. Both datasets were published before the cutoff. Neither records the author, so their bot commits cannot be told apart.

### Settings

C = 0.1 and balanced class weights, chosen with `eval/tune_c.py` on the
mined training projects only: C = 0.1 beat 0.3 on the newest 10% of training
commits, and balanced weights edged out unweighted ones in average recall per
type on six training projects held out from the rest (aws-cdk, commitizen,
element-plus, immich, pnpm and starship).

Calibration teaches the classifier how common each type is, and even with
balanced weights the trained model put `fix` first for most diffs: on the
recent commits of the unseen projects it put `fix` first for 71.1% of
them (only 44.1% are `fix`), and `refactor`, `perf` and `style`
commits almost never got their own type first. It scored 53.5% top-1,
85.8% top-3 and 29.1% macro recall there. `eval/tune_prior.py`
divides each type's probability by its share of the training data raised to a
power alpha, by shifting the calibration intercepts, so the result is still a
plain scikit-learn model.

Alpha has to suit projects the model has not seen: there it is least sure of
itself, so the correction weighs most. A first choice, alpha = 1, made on
`test_seen_recent` (projects the model had seen), over-corrected on the unseen
projects: the first suggestion fell to 39.7% and was `fix` for only
19.1% of the commits. So alpha was chosen again on unseen projects
without touching the test sets. `eval/holdout_split.py` held six training
projects out (aws-cdk, commitizen, element-plus, immich, pnpm and starship,
with their organizations' commits in the public datasets), a model was trained
on the rest, and alpha maximizes the average of top-1 and macro recall on the
six:

| Alpha | Top-1 | Top-3 | Macro recall | Average of top-1 and macro recall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 39.9% | 81.6% | 24.9% | 32.4% |
| 0.1 | 40.2% | 81.8% | 26.5% | 33.4% |
| 0.2 | 40.4% | 82.0% | 28.0% | 34.2% |
| 0.25 | 40.6% | 82.2% | 28.5% | 34.5% |
| 0.3 | 40.8% | 82.4% | 29.0% | 34.9% |
| 0.4 | 41.2% | 82.6% | 30.4% | 35.8% |
| 0.5 | 42.0% | 82.8% | 31.9% | 37.0% |
| 0.6 | 43.1% | 83.2% | 33.8% | 38.5% |
| 0.7 | 44.2% | 83.3% | 35.4% | 39.8% |
| 0.75 | 44.4% | 83.3% | 36.3% | 40.3% |
| 0.8 | 44.5% | 83.2% | 37.2% | 40.9% |
| **0.9** | **43.2%** | **83.1%** | **38.7%** | **41.0%** |
| 1 | 39.6% | 81.4% | 39.5% | 39.5% |

The shipped model, trained on everything, uses alpha = 0.9. Because the
first choice was revised after seeing its result on `test_unseen_recent`,
that set shaped the method, though not the value; `test_unseen_older` was
scored only once, with the final model.

## Results

Commits written by people:

| Set | Commits | Top-1 | Top-3 | Macro recall | Always the most common type |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unseen projects, after training | 10,005 | **43.5%** | **85.9%** | 42.9% | 44.1% (`fix`) |
| Unseen projects, older history | 91,617 | 44.6% | 83.7% | 36.2% | 35.2% (`fix`) |
| Training projects, after training | 39,591 | 58.0% | 89.7% | 44.9% | 39.2% (`fix`) |

Top-1 is how often the first suggestion is the type the author chose; top-3
is how often it is among the three gca lists (picking 3 of the 11 types at
random would give 27.3%). Macro recall is the first suggestion's recall
averaged over the types with at least 20 commits: a model that nearly always
suggests `fix` gets a good top-1 from `fix` being common, but a poor macro
recall.

The previous model on the same sets (`results_previous.json`, the model embedded in the last release; re-exporting its committed joblib gives a slightly weaker model):

| Set | Top-1, previous → this | Top-3, previous → this | Macro recall, previous → this |
| --- | ---: | ---: | ---: |
| Projects never seen, after training | 42.0% → 43.5% | 75.1% → 85.9% | 44.1% → 42.9% |
| … only the 8 projects the previous model never saw either | 33.6% → 45.9% | 68.1% → 86.8% | 42.6% → 47.4% |
| Projects never seen, older history | 41.6% → 44.6% | 73.3% → 83.7% | 41.7% → 36.2% |
| … only the 8 projects the previous model never saw either | 32.2% → 46.4% | 65.7% → 84.8% | 37.4% → 34.6% |
| Projects seen in training, after training | 33.5% → 58.0% | 65.2% → 89.7% | 37.5% → 44.9% |

The previous model's training data included commits from 10 of the 18 test projects (among them quasar, electron, vue core and nest), so for it they were not unseen, and on their older history it was partly scored on commits it had trained on. The 8 projects that neither model saw (hoppscotch, napi-rs, novu, rolldown, shadcn/ui, vitest, vueuse, zitadel) are the fair comparison: there the new model's first suggestion and top three are far better. Average recall per type is where the previous model holds up best, on older history in particular: it put the rarer types first more often, at the cost of being wrong more often overall.

Per-type precision and recall, per-repository results and the bot commits are
in `results.json`; the previous model's scores on the same sets are in
`results_previous.json`.

## With the subject

When the subject is known before the type (`gca -m`, or a subject draft), a
second model reads it: logistic regression over TF-IDF weighted words and
pairs of neighbouring words (30,000 of them), trained on the subjects of the
same 437,945 commits as the diff model, without their type prefix or a
trailing pull request number (`train_subject.py`). gca multiplies the two
models' probabilities:

    p ∝ p_diff · p_subject^weight / prior^prior_power

On the six held-out training projects, with a diff model trained without them,
the settings compared were unigrams (20,000) against unigrams and pairs
(30,000 or 60,000), C from 0.5 to 8, and balanced class weights;
unweighted, 30,000 terms and C = 0.5 did best. `eval/tune_fusion.py` then
scored weights and prior powers (37,278 commits by people; first suggestion
right / right type in top 3 / average recall per type):

| | Top-1 | Top-3 | Macro recall |
| --- | ---: | ---: | ---: |
| Diff only | 43.2% | 83.1% | 38.7% |
| Subject only | 52.3% | 85.5% | 32.6% |
| weight 0.2, prior power 0.15 | 55.8% | 88.3% | 47.9% |
| **weight 0.25, prior power 0.15** | **56.7%** | **88.6%** | **47.2%** |
| weight 0.3, prior power 0.15 | 56.6% | 88.8% | 45.5% |
| weight 0.5, prior power 0.15 | 55.7% | 88.7% | 41.2% |
| weight 0.25, prior power 0 | 56.1% | 89.5% | 39.5% |
| weight 0.25, prior power 0.3 | 50.7% | 84.2% | 52.1% |

The shipped model uses the best first suggestion there: weight 0.25, prior
power 0.15. On the test sets, with each author's own subject
(`results_subject.json`):

| Set | Diff only | Subject only | Both |
| --- | ---: | ---: | ---: |
| Unseen projects, after training | 43.5% · 85.9% · 42.9% | 58.0% · 86.6% · 36.5% | **61.3% · 90.4% · 50.8%** |
| Unseen projects, older history | 44.6% · 83.7% · 36.2% | 55.1% · 85.7% · 34.8% | 57.9% · 88.4% · 45.5% |
| Training projects, after training | 58.0% · 89.7% · 44.9% | 60.5% · 88.2% · 38.7% | 67.9% · 93.6% · 52.0% |

(top-1 · top-3 · macro recall). The gain is largest where the diff is
weakest: on the first set `refactor` goes from 0.0% to 26.6% recall, `perf`
from 2.7% to 18.4%, `fix` from 39.3% to 71.8%; `test` falls from 71.2% to
58.7%, which the path rule makes up for when only tests changed.

A subject draft also counts as the subject when there is no `-m`. That was
decided after scoring it on the test sets: for the 408 commits with a draft in
the first set, the first suggestion was right for 64.2% instead of 59.6%
(5,729 commits in the second: 65.6% instead of 59.9%; 1,265 in the third:
67.6% instead of 68.1%).

The authors wrote these subjects with the type in front of them, so a subject
typed into gca before choosing a type may say less; the numbers are likely an
upper bound on what the subject adds.

## With the project's history

gca also reads the repository's last 500 commits and counts the types people
gave them (bot commits left out). The ranking is tilted toward that mix:

    p ∝ p · (q / prior)^weight,   q = (counts + 10 · prior) / (commits + 10)

where prior is the mix of types in the training data. A project that uses
types as the training data did keeps its ranking; one with no Conventional
Commits in its history is not affected.

`eval/history_sim.py` rebuilds that history from the datasets: for each
commit, the 500 commits before it in the same repository by commit time. On
the six held-out training projects (`eval/tune_history.py`, with the
held-out diff and subject models; a median of 500 typed commits before each
commit), first suggestion right / right type in top 3 / average recall per
type:

| Weight | Diff | Diff and subject |
| ---: | ---: | ---: |
| none | 43.2% / 83.1% / 38.7% | 56.7% / 88.6% / 47.2% |
| 0.05 | 48.6% / 86.2% / 40.6% | 59.3% / 90.5% / 47.9% |
| 0.1 | **50.5% / 87.2% / 41.3%** | 60.9% / 91.6% / 48.1% |
| 0.15 | 49.9% / 87.2% / 40.6% | 61.9% / 92.2% / 48.1% |
| 0.2 | 48.6% / 87.0% / 39.7% | 62.5% / 92.3% / 48.0% |
| 0.25 | 47.2% / 86.4% / 38.3% | **62.6% / 92.3% / 47.6%** |
| 0.3 | 45.7% / 85.6% / 37.0% | 62.2% / 92.2% / 47.2% |

The shipped weights are the best first suggestion for each: 0.1 without a
subject and 0.25 with one. On the test sets (`results_history.json`; for
`test_seen_recent` the history includes the training projects' older
commits, passed with `--history datasets/train.jsonl`):

| Set | Diff | Diff and history | Diff and subject | Diff, subject and history |
| --- | ---: | ---: | ---: | ---: |
| Unseen projects, after training | 43.5% · 85.9% · 42.9% | 51.1% · 89.9% · 48.0% | 61.3% · 90.4% · 50.8% | **66.9% · 93.7% · 55.1%** |
| Unseen projects, older history | 44.6% · 83.7% · 36.2% | 51.0% · 88.1% · 38.9% | 57.9% · 88.4% · 45.5% | 63.1% · 92.5% · 47.2% |
| Training projects, after training | 58.0% · 89.7% · 44.9% | 58.1% · 91.6% · 46.8% | 67.9% · 93.6% · 52.0% | 68.0% · 94.4% · 53.7% |

The datasets hold only Conventional Commits, so the simulated history has
500 typed commits where a real log's 500 may have fewer; fewer commits move
the ranking less.

## Reproduce

From the repository root, with git and Python 3.11+
(`pip install -r requirements.txt`), after the steps in the README's
Retraining section (or with the shipped model, after `eval/collect.sh` only):

```
bash eval/collect.sh   # needs about 12 GB free: each clone is deleted once mined
bash eval/run.sh       # writes eval/results.json and redraws docs/heldout_accuracy.png
python eval/evaluate_subject.py --sets datasets/test_*.jsonl   # writes eval/results_subject.json
python eval/evaluate_history.py --sets datasets/test_*.jsonl --history datasets/train.jsonl
```

`eval/tune_fusion.py` and `eval/tune_history.py` need the held-out split and
the diff model trained without it (see `eval/holdout_split.py`):

```
python eval/tune_fusion.py --diff-model datasets/holdout/model_v2.joblib --alpha 0.9 \
    --train datasets/holdout/train.jsonl datasets/holdout/external.jsonl \
    --val datasets/holdout/validation.jsonl
python eval/tune_history.py (the same arguments)
```

`python eval/evaluate.py ... --predictions FILE` also writes every commit's
label and top three suggestions, for error analysis. The repositories keep
growing, so a later run finds more recent commits than this one (run on
2026-09-29).

## Caveats

- The label is whatever the author chose, and projects differ: dependency
  updates are `chore` in some and `build` or `fix` in others, so part of the
  error is convention rather than the model.
- The numbers are for the model alone. The CLI's path rule (`docs`, `test`,
  `ci`) only changes which suggestion is pre-selected.
- `revert` and `style` are rare, so their per-type numbers are noisy.
