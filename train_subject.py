#!/usr/bin/env python3
"""Train the subject model: the type a commit's subject line suggests.

Words such as "speed up" or "rename" state an intent the diff often hides.
When the subject is known before the type (`gca -m "..."`, or a subject
draft), gca multiplies this model's probabilities into the diff model's:

    p ∝ p_diff · p_subject^weight / prior^prior_power

The weight and prior power are chosen on projects left out of training
(eval/tune_fusion.py) and stored in the exported model, which the CLI embeds.

The model is multinomial logistic regression over TF-IDF weighted words and
pairs of neighbouring words. It trains on the same commits as the diff model:
bot commits and commits without a diff are skipped.

Usage:
    python train_subject.py --data datasets/train.jsonl datasets/external.jsonl \\
        --out out/subject_model.json --weight 0.25 --prior-power 0.15
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from dedupe import iter_rows

# "fix(scope)!: " at the start of a header
PREFIX = re.compile(r"^[A-Za-z]+(?:\([^()\r\n]*\))?!?:\s*")
# the pull request number GitHub appends when squashing: " (#123)", " (#12, #34)"
PR_SUFFIX = re.compile(r"\s*\(#\d+(?:,\s*#\d+)*\)\s*$")


def subject_of(message):
    """What the author typed as the subject: the first line of the message,
    without the type prefix and without a trailing pull request number.
    gca-rs/src/subject.rs reads subjects the same way."""
    first = (message or "").split("\n", 1)[0].strip()
    first = PREFIX.sub("", first, count=1)
    return PR_SUFFIX.sub("", first).strip()


def training_rows(paths, include_bots=False):
    """(subject, label) of every commit the diff model trains on."""
    for row in iter_rows([Path(p) for p in paths]):
        if row.get("is_bot") and not include_bots:
            continue
        if not isinstance(row.get("label"), str) or not str(row.get("diff_text") or "").strip():
            continue
        yield subject_of(row.get("message")), row["label"]


def fit(subjects, labels, C=0.5, max_features=30000, min_df=3):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df,
                          max_features=max_features)
    X = vec.fit_transform(subjects)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X, labels)
    return vec, clf


def export(vec, clf, label_counts, weight, prior_power, digits=6):
    """The JSON the CLI reads. Weights keep `digits` significant digits."""
    r = lambda v: float(f"{v:.{digits}g}")  # noqa: E731
    classes = [str(c) for c in clf.classes_]
    total = sum(label_counts[c] for c in classes)
    return {
        "schema_version": 1,
        "classes": classes,
        "vocabulary": sorted(vec.vocabulary_, key=vec.vocabulary_.get),
        "idf": [r(v) for v in vec.idf_],
        "token_pattern": vec.token_pattern,
        "lowercase": vec.lowercase,
        "sublinear_tf": vec.sublinear_tf,
        "ngram_range": list(vec.ngram_range),
        "coef": [[r(v) for v in row] for row in clf.coef_],
        "intercept": [r(v) for v in clf.intercept_],
        "fusion": {
            "weight": weight,
            "prior_power": prior_power,
            "prior": [label_counts[c] / total for c in classes],
        },
    }


class Exported:
    """NumPy forward pass over the exported JSON, independent of scikit-learn,
    so the evaluation and the Rust parity fixtures use exactly what ships."""

    def __init__(self, payload):
        self.p = payload
        self.classes = payload["classes"]
        self.index = {t: i for i, t in enumerate(payload["vocabulary"])}
        self.idf = np.asarray(payload["idf"])
        self.coef = np.asarray(payload["coef"])
        self.intercept = np.asarray(payload["intercept"])
        self.token_re = re.compile(payload["token_pattern"])
        f = payload["fusion"]
        self.weight, self.prior_power, self.log_prior = f["weight"], f["prior_power"], np.log(f["prior"])

    def features(self, subject):
        text = subject.lower() if self.p["lowercase"] else subject
        tokens = self.token_re.findall(text)
        lo, hi = self.p["ngram_range"]
        counts = Counter()
        for n in range(lo, hi + 1):
            for i in range(len(tokens) - n + 1):
                j = self.index.get(" ".join(tokens[i:i + n]))
                if j is not None:
                    counts[j] += 1
        x = np.zeros(len(self.idf))
        for j, tf in counts.items():
            x[j] = (1 + np.log(tf) if self.p["sublinear_tf"] else tf) * self.idf[j]
        norm = np.sqrt((x * x).sum())
        return x / norm if norm > 0 else x

    def predict_proba(self, subject):
        """For a subject as subject_of() returns it."""
        s = self.coef @ self.features(subject) + self.intercept
        e = np.exp(s - s.max())
        return e / e.sum()

    def combine(self, diff_probs, subject):
        s = (np.log(np.asarray(diff_probs) + 1e-12) + self.weight * np.log(self.predict_proba(subject) + 1e-12)
             - self.prior_power * self.log_prior)
        e = np.exp(s - s.max())
        return e / e.sum()


# Subjects that exercise the corners of reading a subject, for the fixtures.
EDGE_CASES = [
    "",
    "fix(parser)!: handle CRLF line endings (#123)",
    "Speed UP the diff reader",
    "perf: speed up (#12, #34)",
    "Revert \"feat: add dark mode\"",
    "rename util.rs to helpers.rs",
    "bump zod from 3.22.0 to 3.23.8",
    "İstanbul saat dilimi düzeltmesi",
    "nai\u0308ve cafe\u0301 handling",
    "修正登入時的錯誤",
    "ＦＩＸ full-width letters",
    "launch 🚀 v2 → v3 migration",
    "update_snake_case and camelCase names",
    "a b c d e",
]


def write_fixtures(model_path, data_path, out_path, n=150, seed=0):
    """Cases for gca-rs/tests/parity.rs: raw subjects, as a user would pass
    them to -m, with the probabilities this module computes."""
    model = Exported(json.loads(Path(model_path).read_text(encoding="utf-8")))
    rows = [r for r in iter_rows([Path(data_path)]) if not r.get("is_bot") and r.get("message")]
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    subjects = EDGE_CASES + [rows[i]["message"].split("\n", 1)[0] for i in sorted(picked)]
    cases = []
    for raw in subjects:
        diff = rng.dirichlet(np.ones(len(model.classes)))
        cases.append({"subject": raw, "diff_probs": diff.tolist(),
                      "expected_subject": model.predict_proba(subject_of(raw)).tolist(),
                      "expected_combined": model.combine(diff, subject_of(raw)).tolist()})
    Path(out_path).write_text(json.dumps({"classes": model.classes, "cases": cases}, ensure_ascii=False, indent=1)
                              + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out_path}")


def check_fixtures(model_path, fixtures_path, tol=1e-9):
    """The fixtures still describe the model at model_path."""
    model = Exported(json.loads(Path(model_path).read_text(encoding="utf-8")))
    fixtures = json.loads(Path(fixtures_path).read_text(encoding="utf-8"))
    if fixtures["classes"] != model.classes:
        raise SystemExit("fixture classes differ from the model's")
    gap = 0.0
    for case in fixtures["cases"]:
        subject = subject_of(case["subject"])
        gap = max(gap, float(np.abs(model.predict_proba(subject) - case["expected_subject"]).max()),
                  float(np.abs(model.combine(case["diff_probs"], subject) - case["expected_combined"]).max()))
    if gap > tol:
        raise SystemExit(f"fixtures differ from the model by {gap}; rewrite them with --write-fixtures")
    print(f"{len(fixtures['cases'])} fixture cases match (largest difference {gap:.1e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", help="training JSONL files")
    ap.add_argument("--out", default="out/subject_model.json")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--weight", type=float, help="from eval/tune_fusion.py")
    ap.add_argument("--prior-power", type=float, help="from eval/tune_fusion.py")
    ap.add_argument("--write-fixtures", metavar="JSON",
                    help="instead of training, write parity fixtures for the model at --out")
    ap.add_argument("--fixture-data", metavar="JSONL", help="commits to take fixture subjects from")
    ap.add_argument("--check-fixtures", metavar="JSON",
                    help="instead of training, check parity fixtures against the model at --out")
    args = ap.parse_args()
    if args.check_fixtures:
        check_fixtures(args.out, args.check_fixtures)
        return
    if args.write_fixtures:
        if not args.fixture_data:
            ap.error("--write-fixtures needs --fixture-data")
        write_fixtures(args.out, args.fixture_data, args.write_fixtures)
        return
    if not args.data or args.weight is None or args.prior_power is None:
        ap.error("training needs --data, --weight and --prior-power")

    started = time.time()
    subjects, labels = [], []
    for s, y in training_rows(args.data):
        subjects.append(s)
        labels.append(y)
    counts = Counter(labels)
    print(f"{len(labels)} commits", flush=True)
    vec, clf = fit(subjects, np.asarray(labels), C=args.C, max_features=args.max_features)
    payload = export(vec, clf, counts, args.weight, args.prior_power)

    # The exported (rounded) model must agree with scikit-learn.
    sample = subjects[:: max(1, len(subjects) // 500)][:500]
    exported = Exported(payload)
    ours = np.array([exported.predict_proba(s) for s in sample])
    theirs = clf.predict_proba(vec.transform(sample))
    gap = float(np.abs(ours - theirs).max())
    if gap > 1e-4:
        raise SystemExit(f"exported model disagrees with scikit-learn by {gap}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    report = {"commits": len(labels), "labels": dict(counts.most_common()), "C": args.C,
              "max_features": args.max_features, "weight": args.weight,
              "prior_power": args.prior_power, "max_export_gap": gap}
    out.with_name("subject_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f}s; "
          f"largest gap to scikit-learn {gap:.1e}")


if __name__ == "__main__":
    main()
