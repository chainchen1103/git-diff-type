#!/usr/bin/env python3
"""Score the type model against gca's own ranking on the held-out projects.

For each scored commit (by a person, typed, with a diff) of the two unseen
sets, each ranking as gca builds it (eval/evaluate_history.py): the diff
model alone, then the project's history, the same files' history and the
author's own commits, without and with the subject. The diff model is gca's
built-in linear model (linear), the type model (type model), or both
averaged in log space, half each, as gca does when a type model is set
(both). As in gca, the author's own earlier commits are read again by the
linear model alone.

    python draft_model/eval_type.py [--predictions draft_model/type_predictions]
        [--out draft_model/type_results.json]

predict_type.py writes the predictions. Prints the first suggestion, the top
three, the average recall and the average F1 per type (types with at least
20 commits), and F1 for each type.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from dedupe import iter_rows  # noqa: E402
from evaluate_subject import score  # noqa: E402
from history_sim import (FILE_PSEUDO_COMMITS, counts_before, own_choices, scored, timelines,  # noqa: E402
                         weigh, weigh_own, window_start)
from train_enhanced import FEATURE_COLUMNS, prepare_features  # noqa: E402
from train_subject import Exported, subject_of  # noqa: E402
import evaluate_history as eh  # noqa: E402

SETS = ["test_unseen_recent", "test_unseen_older"]


def normalized(logp):
    m = logp.max(axis=1, keepdims=True)
    return logp - m - np.log(np.exp(logp - m).sum(axis=1, keepdims=True))


def metrics(logp, y, classes):
    first = classes[np.argmax(logp, axis=1)]
    out = score(logp, classes, y)
    per = {}
    for c in classes:
        tp = int(((first == c) & (y == c)).sum())
        pred, sup = int((first == c).sum()), int((y == c).sum())
        prec, rec = (tp / pred if pred else 0.0), (tp / sup if sup else 0.0)
        per[str(c)] = {"support": sup, "precision": prec, "recall": rec,
                       "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    out["macro_f1"] = float(np.mean([v["f1"] for v in per.values() if v["support"] >= 20]))
    out["per_type"] = per
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=str(ROOT / "datasets"))
    ap.add_argument("--predictions", default=str(HERE / "type_predictions"))
    ap.add_argument("--diff-model", default=str(ROOT / "out/model_v2.joblib"))
    ap.add_argument("--subject-model", default=str(ROOT / "out/subject_model.json"))
    ap.add_argument("--out", default=str(HERE / "type_results.json"))
    args = ap.parse_args()
    data = Path(args.datasets)
    diff_model = joblib.load(args.diff_model)
    subject_model = Exported(json.loads(Path(args.subject_model).read_text(encoding="utf-8")))
    classes = np.array(subject_model.classes)
    labels = json.loads((Path(args.diff_model).parent / "train_report.json").read_text())["labels"]
    prior = np.array([labels[c] for c in classes], dtype=float)
    prior /= prior.sum()
    by_repo = timelines([str(data / f"{n}.jsonl") for n in SETS])

    sets, pool = {}, {"keys": [], "times": [], "labels": []}
    for name in SETS:
        rows = [r for r in iter_rows([data / f"{name}.jsonl"]) if scored(r)]
        lin = np.vstack([diff_model.predict_proba(prepare_features(pd.DataFrame(rows[k:k + 5000]), 20000)
                                                  [FEATURE_COLUMNS]) for k in range(0, len(rows), 5000)])
        p = {}
        with open(Path(args.predictions) / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p[(r.get("repo"), r["id"])] = [r["p"][c] for c in classes]
        missing = sum((r.get("repo"), r.get("sha")) not in p for r in rows)
        if missing:
            raise SystemExit(f"{name}: {missing} commits without a prediction")
        nn = np.array([p[(r.get("repo"), r.get("sha"))] for r in rows])
        n, same = counts_before(by_repo, rows, classes)
        first = len(pool["keys"])
        pool["keys"] += [(r.get("repo"), r.get("author") or "") for r in rows]
        pool["times"] += [r.get("committed_at") or "" for r in rows]
        pool["labels"] += [r["label"] for r in rows]
        subj = np.log(np.array([subject_model.predict_proba(subject_of(r.get("message"))) for r in rows]) + 1e-12)
        sets[name] = {"targets": range(first, first + len(rows)), "y": np.array([r["label"] for r in rows]),
                      "lin": np.log(lin + 1e-12), "nn": np.log(nn + 1e-12), "subject": subj, "n": n,
                      "same": same, "starts": window_start(by_repo, rows)}
        print(f"{name}: {len(rows)} commits", flush=True)

    def fuse(logp, s):
        return normalized(logp + subject_model.weight * s["subject"]
                          - subject_model.prior_power * subject_model.log_prior)

    # what gca showed for each commit of the pool, from the linear model
    shown = {"diff": np.exp(np.vstack([sets[n]["lin"] for n in SETS])),
             "both": np.exp(np.vstack([fuse(sets[n]["lin"], sets[n]) for n in SETS]))}
    results = {}
    line = lambda v: f"{v['top1']:.1%} · {v['top3']:.1%} · {v['macro_recall']:.1%} · {v['macro_f1']:.1%}"
    for name, s in sets.items():
        own = {k: own_choices(pool["keys"], pool["times"], pool["labels"], shown[k], classes, s["targets"],
                              s["starts"]) for k in ("diff", "both")}
        models = {"linear": s["lin"], "type model": normalized(s["nn"]),
                  "both": normalized(0.5 * s["nn"] + 0.5 * s["lin"])}
        results[name] = {}
        for model, logp in models.items():
            both = fuse(logp, s)
            files = weigh(weigh(logp, s["n"], prior, eh.WEIGHT), s["same"], prior, eh.FILE_WEIGHT,
                          FILE_PSEUDO_COMMITS)
            both_files = weigh(weigh(both, s["n"], prior, eh.WEIGHT_WITH_SUBJECT), s["same"], prior,
                               eh.FILE_WEIGHT_WITH_SUBJECT, FILE_PSEUDO_COMMITS)
            rankings = {
                "alone": logp,
                "history_own": weigh_own(files, *own["diff"], prior, eh.OWN_WEIGHT),
                "subject": both,
                "subject_history_own": weigh_own(both_files, *own["both"], prior, eh.OWN_WEIGHT_WITH_SUBJECT),
            }
            results[name][model] = {r: metrics(lp, s["y"], classes) for r, lp in rankings.items()}
        print(f"\n{name} (first suggestion · top three · average recall · average F1)")
        print("| Model | Alone | With history and own commits | With the subject | Subject, history and own |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for model, r in results[name].items():
            print(f"| {model} | " + " | ".join(line(r[k]) for k in ("alone", "history_own", "subject",
                                                                    "subject_history_own")) + " |")
        print("\nF1 per type, with history and own commits / also with the subject:")
        for t in classes:
            a, b = (results[name][m]["history_own"]["per_type"][t] for m in ("linear", "both"))
            c, d = (results[name][m]["subject_history_own"]["per_type"][t] for m in ("linear", "both"))
            if a["support"] >= 20:
                print(f"  {t:9} {a['support']:6}  {a['f1']:.0%} -> {b['f1']:.0%}   {c['f1']:.0%} -> {d['f1']:.0%}")
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
