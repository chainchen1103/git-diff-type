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

Besides the words of the diff, its file paths and extensions, the overlap
between added and removed words and the line counts, the model reads nine
signs of whether the change alters behavior (`behavior_features` in
`train_enhanced.py`, mirrored in `gca-rs/src/features.rs`): the shares of
changed lines that were only moved, only renamed or changed in literals,
only reformatted, or are comments; the shares of files that are new,
deleted, renamed or tests; and how many words about performance ("cache",
"lazy", "batch" and the like) the added lines have. gca 0.5 added them; see
[Behavior features](#behavior-features) for what they changed.

Calibration teaches the classifier how common each type is, and even with
balanced weights the trained model put `fix` first for most diffs: on the
recent commits of the unseen projects it put `fix` first for 65.6% of
them (only 44.1% are `fix`), and `refactor`, `perf` and `style`
commits never got their own type first. It scored 55.1% top-1,
86.8% top-3 and 33.0% macro recall there. `eval/tune_prior.py`
divides each type's probability by its share of the training data raised to a
power alpha, by shifting the calibration intercepts, so the result is still a
plain scikit-learn model.

Alpha has to suit projects the model has not seen: there it is least sure of
itself, so the correction weighs most. When gca 0.2 introduced the
correction, a first choice, alpha = 1, made on `test_seen_recent` (projects
the model had seen), over-corrected on the unseen projects: the first
suggestion fell to 39.7% and was `fix` for only 19.1% of the commits. So
alpha is chosen on unseen projects without touching the test sets.
`eval/holdout_split.py` held six training projects out (aws-cdk, commitizen,
element-plus, immich, pnpm and starship, with their organizations' commits in
the public datasets), a model was trained on the rest, and alpha maximizes
the average of top-1 and macro recall on the six:

| Alpha | Top-1 | Top-3 | Macro recall | Average of top-1 and macro recall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 41.6% | 82.4% | 27.5% | 34.5% |
| 0.1 | 42.3% | 82.6% | 29.5% | 35.9% |
| 0.2 | 43.0% | 82.8% | 31.0% | 37.0% |
| 0.25 | 43.4% | 82.9% | 31.6% | 37.5% |
| 0.3 | 43.8% | 83.0% | 32.4% | 38.1% |
| 0.4 | 44.8% | 83.3% | 34.1% | 39.5% |
| 0.5 | 45.7% | 83.6% | 35.4% | 40.5% |
| 0.6 | 46.8% | 83.8% | 36.8% | 41.8% |
| 0.7 | 47.2% | 84.0% | 38.1% | 42.6% |
| 0.75 | 47.0% | 84.0% | 38.7% | 42.8% |
| 0.8 | 46.4% | 84.1% | 39.3% | 42.8% |
| 0.85 | 45.6% | 83.9% | 39.9% | 42.7% |
| **0.9** | **44.7%** | **83.8%** | **40.5%** | **42.6%** |
| 0.95 | 43.5% | 83.4% | 41.1% | 42.3% |
| 1 | 41.4% | 82.2% | 41.7% | 41.6% |

gca 0.2 to 0.4 used alpha = 0.9, the best for their model. For this one
0.75 is the best by that measure, 0.2 points ahead of 0.9, but the shipped
model, trained on everything, keeps 0.9; [Behavior features](#behavior-features)
says why.

## Behavior features

gca 0.5 added the nine behavior features and retrained the model on the same
commits. On the six held-out training projects, with every setting of gca 0.4
(alpha 0.9, the subject's weight and prior power, the history's weights), a
model trained without them, as in gca 0.4, against one trained with them
(first suggestion right / right type in top 3 / average recall per type;
"history" is the project's mix, the same files and the author's own commits,
as below):

| Ranking | gca 0.4 | gca 0.5 |
| --- | ---: | ---: |
| Diff | 43.2% / 83.1% / 38.7% | 44.7% / 83.8% / 40.5% |
| Diff and subject | 56.7% / 88.6% / 47.2% | 57.0% / 88.8% / 47.8% |
| Diff and history | 57.2% / 91.0% / 44.0% | 58.2% / 91.5% / 45.3% |
| Diff, subject and history | 66.3% / 93.7% / 49.3% | 66.7% / 93.8% / 50.0% |

Tuned again one setting at a time, as for gca 0.4, the new model would have
taken alpha 0.75 (above) and, after it, a weight of 0.15 for the project's
mix and 0.2 for the same files with a subject. A model built that way was
scored on the test sets first. It was right first more often (from the diff
alone, 50.3% instead of 43.5% on the recent commits of the unseen projects),
but put `perf`, `refactor`, `style` and `build` first less often than gca 0.4,
even with the subject and the whole history (there, `perf` commits got `perf`
first 31.4% of the time instead of 42.0%), and its average recall per type
fell in five of the six rankings with the whole history. The held-out
projects already showed it: with the history in the ranking, which is what
gca shows in most repositories, a smaller alpha gains few first suggestions
and loses average recall per type (history weights tuned again for each
alpha; first suggestion right / average recall per type / their average):

| Alpha | Diff and history | Diff, subject and history |
| ---: | ---: | ---: |
| 0.75 | 58.3% / 42.7% / 50.5% | 66.9% / 49.0% / 58.0% |
| 0.8 | 58.9% / 44.0% / 51.5% | 67.0% / 49.6% / 58.3% |
| 0.85 | 58.5% / 44.7% / 51.6% | 66.8% / 50.1% / 58.5% |
| **0.9** | **58.2% / 45.3% / 51.7%** | **66.7% / 50.0% / 58.4%** |
| 0.95 | 57.6% / 45.9% / 51.7% | 67.2% / 50.8% / 59.0% |
| 1 | 56.8% / 46.1% / 51.5% | 67.0% / 51.1% / 59.0% |
| gca 0.4, alpha 0.9 | 57.2% / 44.0% / 50.6% | 66.3% / 49.3% / 57.8% |

From 0.85 to 0.95 the average without a subject is within 0.2 points; with
one it keeps rising, while the diff alone loses more (its first suggestion
is right 43.5% of the time at 0.95, 44.7% at 0.9). gca 0.5 keeps alpha 0.9
and every other setting of 0.4, with which the new model beats gca 0.4's on
every measure in the first table, and on the test sets (below); tuned again
at 0.9, the history's weights come out as they were. Only the subject's
prior power would move, to 0.1, for half a point more first suggestions
right and 2.9 points less average recall per type
([With the subject](#with-the-subject)); it stays 0.15. The test sets shaped
this choice, since the first model had been scored on them.

## Results

Commits written by people:

| Set | Commits | Top-1 | Top-3 | Macro recall | Always the most common type |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unseen projects, after training | 10,005 | **45.5%** | **86.5%** | 44.3% | 44.1% (`fix`) |
| Unseen projects, older history | 91,617 | 45.9% | 84.3% | 36.9% | 35.2% (`fix`) |
| Training projects, after training | 39,591 | 58.8% | 90.2% | 46.1% | 39.2% (`fix`) |

Top-1 is how often the first suggestion is the type the author chose; top-3
is how often it is among the three gca lists (picking 3 of the 11 types at
random would give 27.3%). Macro recall is the first suggestion's recall
averaged over the types with at least 20 commits: a model that nearly always
suggests `fix` gets a good top-1 from `fix` being common, but a poor macro
recall.

gca 0.2 to 0.4 shipped a model trained on the same commits without the
behavior features, with alpha = 0.9 (`results_previous.json`):

| Set | Top-1, 0.4 → 0.5 | Top-3, 0.4 → 0.5 | Macro recall, 0.4 → 0.5 |
| --- | ---: | ---: | ---: |
| Unseen projects, after training | 43.5% → 45.5% | 85.9% → 86.5% | 42.9% → 44.3% |
| Unseen projects, older history | 44.6% → 45.9% | 83.7% → 84.3% | 36.2% → 36.9% |
| Training projects, after training | 58.0% → 58.8% | 89.7% → 90.2% | 44.9% → 46.1% |

On the first set, `test` commits got `test` first 79.4% of the time instead
of 71.2% (and `test` was right 29.5% of the times it came first, instead of
21.4%), `feat` 45.4% instead of 42.7%, `perf` 4.7% instead of 2.7%; the right
type was among the three for 26.3% of `refactor` commits instead of 23.4%,
and for 25.9% of `perf` commits instead of 22.4%. `build` (118 commits) and
`ci` lost a little (28.8% instead of 33.1%, 76.3% instead of 77.9%), and
`refactor` is still never first from the diff alone.

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
| Diff only | 44.7% | 83.8% | 40.5% |
| Subject only | 52.3% | 85.5% | 32.6% |
| weight 0.2, prior power 0.15 | 56.1% | 88.5% | 48.9% |
| **weight 0.25, prior power 0.15** | **57.0%** | **88.8%** | **47.8%** |
| weight 0.3, prior power 0.15 | 57.4% | 89.0% | 46.3% |
| weight 0.5, prior power 0.15 | 56.1% | 88.9% | 41.8% |
| weight 0.25, prior power 0 | 56.9% | 89.5% | 40.3% |
| weight 0.25, prior power 0.1 | 57.5% | 89.2% | 44.9% |
| weight 0.25, prior power 0.3 | 51.3% | 84.9% | 53.0% |

gca uses weight 0.25 and prior power 0.15, the best first suggestion with gca
0.4's diff model. With this one, prior power 0.1 is half a point ahead in
first suggestions and 2.9 points behind in average recall per type, and
weight 0.3 is 0.4 points ahead and 1.5 behind; gca keeps its settings (see
[Behavior features](#behavior-features)). On the test sets, with each
author's own subject (`results_subject.json`):

| Set | Diff only | Subject only | Both |
| --- | ---: | ---: | ---: |
| Unseen projects, after training | 45.5% · 86.5% · 44.3% | 58.0% · 86.6% · 36.5% | **61.7% · 90.8% · 51.6%** |
| Unseen projects, older history | 45.9% · 84.3% · 36.9% | 55.1% · 85.7% · 34.8% | 58.1% · 88.6% · 45.8% |
| Training projects, after training | 58.8% · 90.2% · 46.1% | 60.5% · 88.2% · 38.7% | 68.3% · 93.9% · 52.7% |

(top-1 · top-3 · macro recall). The gain is largest where the diff is
weakest: on the first set `refactor` goes from 0.0% to 26.1% recall, `perf`
from 4.7% to 20.0%, `fix` from 40.7% to 71.5%; `test` falls from 79.4% to
64.0%, which the path rule makes up for when only tests changed. With gca
0.4's diff model, both together scored 61.3% · 90.4% · 50.8% there.

A subject draft also counts as the subject when there is no `-m`. That was
decided after scoring it on the test sets: for the 408 commits with a draft in
the first set, the first suggestion was right for 64.0% instead of 60.8%
(5,729 commits in the second: 66.0% instead of 60.8%; 1,265 in the third:
68.6% instead of 70.1%).

The authors wrote these subjects with the type in front of them, so a subject
typed into gca before choosing a type may say less; the numbers are likely an
upper bound on what the subject adds.

## With the project's history

gca also reads the repository's last 500 commits and counts the types people
gave them (bot commits left out). The ranking is tilted toward that mix:

    p ∝ p · (q / prior)^weight,   q = (counts + 10 · prior) / (commits + 10)

where prior is the mix of types in the training data. A project that uses
types as the training data did keeps its ranking; one with no Conventional
Commits in its history is not affected. The ranking is then tilted the same
way toward the mix of the commits among those 500 that touched a file being
committed, smoothed with 5 commits' worth of the prior instead of 10.

`eval/history_sim.py` rebuilds that history from the datasets: for each
commit, the 500 commits before it in the same repository by commit time, and
the files each of them touched, read from its diff. On the six held-out
training projects (`eval/tune_history.py`, with the held-out diff and
subject models; a median of 500 typed commits before each commit), first
suggestion right / right type in top 3 / average recall per type:

| Weight | Diff | Diff and subject |
| ---: | ---: | ---: |
| none | 44.7% / 83.8% / 40.5% | 57.0% / 88.8% / 47.8% |
| 0.05 | 49.6% / 86.5% / 42.3% | 59.4% / 90.6% / 48.6% |
| 0.1 | **51.7% / 87.6% / 43.1%** | 61.0% / 91.7% / 49.0% |
| 0.15 | 51.2% / 87.9% / 42.4% | 62.3% / 92.2% / 49.1% |
| 0.2 | 49.9% / 87.8% / 41.3% | 62.8% / 92.4% / 48.9% |
| 0.25 | 48.6% / 87.3% / 40.1% | **62.9% / 92.4% / 48.7%** |
| 0.3 | 47.0% / 86.6% / 38.6% | 62.6% / 92.4% / 48.3% |

The shipped weights are the best first suggestion for each, as with gca
0.4's diff model: 0.1 without a subject and 0.25 with one. With them, the
same files' mix (86.5% of these commits have earlier commits to the same
files, a median of 6):

| Weight | Diff | Diff and subject |
| ---: | ---: | ---: |
| none | 51.7% / 87.6% / 43.1% | 62.9% / 92.4% / 48.7% |
| 0.05 | 54.0% / 88.9% / 44.8% | 64.0% / 92.8% / 50.0% |
| 0.1 | **54.5% / 89.6% / 45.8%** | 64.7% / 93.0% / 50.9% |
| 0.15 | 54.1% / 89.7% / 46.5% | **65.1% / 93.1% / 51.3%** |
| 0.2 | 53.3% / 89.5% / 46.9% | 64.9% / 93.1% / 51.7% |
| 0.3 | 51.2% / 89.1% / 46.5% | 64.1% / 92.9% / 52.0% |

The shipped weights are again the best first suggestion for each: 0.1
without a subject and 0.15 with one. The smoothing matters little: with 2 or
10 commits' worth instead of 5, the best first suggestion is 54.3% either way
without a subject and 65.1% or 65.0% with one.

On the test sets (`results_history.json`; for `test_seen_recent` the history
includes the training projects' older commits, passed with
`--history datasets/train.jsonl`), each with the project's mix and then the
same files' mix too:

| Set | Diff | + project | + same files | Diff and subject | + project | + same files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unseen projects, after training | 45.5% · 86.5% · 44.3% | 53.8% · 90.1% · 49.9% | 58.0% · 91.5% · 55.1% | 61.7% · 90.8% · 51.6% | 67.2% · 94.1% · 55.9% | **68.3% · 94.6% · 59.5%** |
| Unseen projects, older history | 45.9% · 84.3% · 36.9% | 51.9% · 88.6% · 39.5% | 55.1% · 90.2% · 42.8% | 58.1% · 88.6% · 45.8% | 63.2% · 92.6% · 47.4% | 64.8% · 93.5% · 50.5% |
| Training projects, after training | 58.8% · 90.2% · 46.1% | 59.2% · 92.2% · 48.1% | 60.8% · 93.0% · 51.3% | 68.3% · 93.9% · 52.7% | 68.4% · 94.6% · 54.2% | 68.9% · 95.0% · 55.7% |

82.4%, 88.7% and 77.7% of those commits have earlier commits to the same
files. The datasets hold only Conventional Commits, so the simulated history
has 500 typed commits where a real log's 500 may have fewer; fewer commits
move the ranking less. The file lists come from the stored diffs, which the
miner cuts at 20,000 characters, so a large commit's later files are missing
from the simulation; gca reads the full lists from git.

## With your own commits

Last, gca takes the user's latest 10 commits among those 500 (the same email
or name as `git var GIT_AUTHOR_IDENT`), reads each again with the models
(the diff model, and the subject model too when the ranking has a subject),
and compares:

    p ∝ p · ((chosen + prior) / (shown + prior))^weight

where chosen counts the types the user gave those commits and shown sums the
probabilities the models give them, both smoothed with one commit's worth of
the training mix. A type the user chose more often than the models
suggested it rises. The simulation lets each commit's author stand for the
user: the models read the author's latest 10 commits among the 500 before
it. `gca-rs` reads those commits' patches as the miner does, cut at 20,000
characters and with the same line counts.

On the six held-out training projects, with the weights above for the
project's mix and the same files (85.3% of these commits have earlier
commits by their author):

| Weight | Diff | Diff and subject |
| ---: | ---: | ---: |
| none | 54.5% / 89.6% / 45.8% | 65.1% / 93.1% / 51.3% |
| 0.05 | 57.6% / 91.3% / 46.1% | 66.1% / 93.7% / 50.8% |
| 0.1 | **58.2% / 91.5% / 45.3%** | 66.5% / 93.9% / 50.6% |
| 0.15 | 57.7% / 91.3% / 44.2% | **66.7% / 93.8% / 50.0%** |
| 0.2 | 57.0% / 90.9% / 42.9% | 66.4% / 93.6% / 49.1% |
| 0.25 | 56.2% / 90.5% / 41.7% | 65.9% / 93.4% / 48.3% |
| 0.3 | 55.5% / 90.0% / 41.0% | 65.2% / 93.1% / 47.5% |

The shipped weights are the best first suggestion for each, as with gca
0.4's diff model: 0.1 without a subject and 0.15 with one (ahead of 0.1 by
0.17 points). Of the latest 5, 10 or 20 commits, smoothed with 1, 2, 5 or 10
commits' worth, 10 with 1 gives the best first suggestion without a subject
and is 0.02 points short of the best with one (10 with 5). On the test sets,
on top of the project's mix and the same files (`results_history.json`):

| Set | Diff and history | + own commits | Diff, subject and history | + own commits |
| --- | ---: | ---: | ---: | ---: |
| Unseen projects, after training | 58.0% · 91.5% · 55.1% | **63.8% · 93.0% · 56.2%** | 68.3% · 94.6% · 59.5% | **70.8% · 94.9% · 60.3%** |
| Unseen projects, older history | 55.1% · 90.2% · 42.8% | 60.8% · 92.1% · 44.6% | 64.8% · 93.5% · 50.5% | 67.9% · 94.6% · 50.5% |
| Training projects, after training | 60.8% · 93.0% · 51.3% | 64.8% · 93.7% · 51.5% | 68.9% · 95.0% · 55.7% | 70.8% · 95.2% · 55.5% |

87.2%, 88.7% and 91.2% of those commits have earlier commits by their author.
For `test_seen_recent`, the models read the authors' commits in the training
data again, as the shipped model would read a user's older commits; it was
trained on them. The datasets name authors, not emails, so the simulation
matches names.

With gca 0.4's diff model, the same rankings got 62.3% · 92.5% · 54.4% and
70.2% · 94.6% · 59.2% on the first set (`results_history.json` in 0.4); gca
0.5 is ahead in every cell of this table and of the one in
[With the project's history](#with-the-projects-history).

## Scope suggestions

In a project that uses scopes (at least five of the last 500 commits by
people follow Conventional Commits, and one in five of those has a scope),
gca pre-fills the scope prompt. gca 0.3 suggested the scope used most often
by the commits to the same files, or else to other files in the same
directories, and counted only commits with a scope, so it nearly always
pre-filled one. Now a commit without a scope is a vote for none, commits
sharing the deepest directory with the change come after those to the same
files, and the user's own commits count several times
(`gca-rs/src/history.rs`). `eval/evaluate_scope.py` scores both, with each
commit's author standing for the user: how often the prompt is pre-filled
exactly as the commit has it (empty when it has no scope), how often a scope
is pre-filled at all, and how often that one is right.

On the six held-out training projects, by how many times the user's own
commits count:

| Suggestion | Right | Scope pre-filled | Right when pre-filled |
| --- | ---: | ---: | ---: |
| gca 0.3 | 47.0% | 88.4% | 47.7% |
| 1× | 62.1% | 56.8% | 65.7% |
| 2× | 62.8% | 57.3% | 66.1% |
| 3× | 63.1% | 57.7% | 66.0% |
| 4× | 63.2% | 57.9% | 66.0% |
| **8×** | **63.3%** | 58.5% | 65.6% |
| 16× | 63.3% | 58.9% | 65.3% |

gca counts them 8 times, the best (16 times is 0.003 points behind). On the
test sets (`results_scope.json`; for `test_seen_recent` the history
includes the training projects' older commits):

| Set | Asked | gca 0.3 | Now |
| --- | ---: | ---: | ---: |
| Unseen projects, after training | 92.6% | 42.1% · 89.8% · 42.9% | **52.7% · 69.6% · 52.2%** |
| Unseen projects, older history | 75.2% | 38.7% · 86.7% · 33.7% | 63.6% · 47.4% · 54.8% |
| Training projects, after training | 98.3% | 45.5% · 87.3% · 48.0% | 56.1% · 78.0% · 55.9% |

(right · scope pre-filled · right when pre-filled; "asked" is the share of
commits in a project that used scopes then). 77.9%, 55.6% and 82.5% of the
commits asked about have a scope. As with the types, the simulated history
holds only Conventional Commits.

## Reproduce

From the repository root, with git and Python 3.11+
(`pip install -r requirements.txt`), after the steps in the README's
Retraining section (or with the shipped model, after `eval/collect.sh` only):

```
bash eval/collect.sh   # needs about 12 GB free: each clone is deleted once mined
bash eval/run.sh       # writes eval/results.json and redraws docs/heldout_accuracy.png
python eval/evaluate_subject.py --sets datasets/test_*.jsonl   # writes eval/results_subject.json
python eval/evaluate_history.py --sets datasets/test_*.jsonl --history datasets/train.jsonl
python eval/evaluate_scope.py --sets datasets/test_*.jsonl --history datasets/train.jsonl
```

`eval/tune_fusion.py` and `eval/tune_history.py` need the held-out split and
the diff model trained without it (see `eval/holdout_split.py`):

```
python eval/tune_fusion.py --diff-model datasets/holdout/model_v2.joblib --alpha 0.9 \
    --train datasets/holdout/train.jsonl datasets/holdout/external.jsonl \
    --val datasets/holdout/validation.jsonl
python eval/tune_history.py (the same arguments)
python eval/evaluate_scope.py --sets datasets/holdout/validation.jsonl --own-votes 1 2 3 4 8 16 \
    --out scope_holdout.json
```

The comparisons under [Behavior features](#behavior-features) are
`eval/tune_history.py` runs with each `--alpha`, and the same with a diff
model trained by gca 0.4's `train_enhanced.py`.

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
