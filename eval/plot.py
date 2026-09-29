#!/usr/bin/env python3
"""Draw the README's per-type chart (docs/heldout_accuracy.png) and the
confusion matrix (out/confusion_matrix.png) from eval/results.json.

Usage (from the repository root, after eval/run.sh):
    python eval/plot.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LIGHT, DARK = "#86b6ef", "#1c5cab"  # one hue, light and dark


def chart_text(results, split):
    """The chart's words and numbers: title, subtitle, one row per type
    (fewest commits first), legend and footnote."""
    res = results["test_unseen_recent"]["humans"]
    per_type = {k: v for k, v in res["per_type"].items() if v["support"] >= 20}
    base = res["baseline_most_common"]
    rows = [(f"{k}  (n={per_type[k]['support']:,})", per_type[k]["recall"], per_type[k]["top3_recall"])
            for k in sorted(per_type, key=lambda k: per_type[k]["support"])]
    return {
        "title": f"gca on {res['commits']:,} commits from projects it never saw",
        "subtitle": (f"First suggestion right: {res['top1'] * 100:.1f}%   ·   right type in the top 3: "
                     f"{res['top3'] * 100:.1f}%   ·   always guessing '{base['type']}': "
                     f"{base['top1'] * 100:.1f}%"),
        "rows": rows,
        "legend": ["1st suggestion correct", "correct type in top 3"],
        "footnote": (f"Commits written by people in 18 projects never used in training, landed on or after "
                     f"{split['cutoff']}. Types with fewer than 20 commits omitted. Reproduce: eval/README.md"),
    }


def draw(text, out):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (_, top1, top3) in enumerate(text["rows"]):
        ax.barh(i, top3, height=0.56, color=LIGHT, zorder=2)
        ax.barh(i, top1, height=0.56, color=DARK, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.text(top3 + 0.012, i, f"{top1 * 100:.0f}% · {top3 * 100:.0f}%",
                va="center", ha="left", fontsize=13, color=INK2)
    ax.set_yticks(range(len(text["rows"])))
    ax.set_yticklabels([label for label, _, _ in text["rows"]], fontsize=14, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=12)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.035, 0.94, text["title"], fontsize=24, color=INK, weight="bold")
    fig.text(0.035, 0.895, text["subtitle"], fontsize=15, color=INK2)
    ax.legend(handles=[Patch(color=DARK, label=text["legend"][0]), Patch(color=LIGHT, label=text["legend"][1])],
              loc="lower right", frameon=False, fontsize=13, labelcolor=INK2)
    fig.text(0.035, 0.03, text["footnote"], fontsize=11, color=MUTED)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.1)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def draw_confusion(results, out):
    """Rows: the type the author chose; columns: gca's first suggestion. Each
    row is shaded by its share, so every type reads the same way whatever
    its size."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    res = results["test_unseen_recent"]["humans"]
    confusion = res.get("confusion")
    if not confusion:
        return False
    types = sorted(res["per_type"], key=lambda k: -res["per_type"][k]["support"])
    counts = np.array([[confusion.get(a, {}).get(b, 0) for b in types] for a in types], dtype=float)
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, LIGHT, DARK])
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 10.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(types)):
        for j in range(len(types)):
            if counts[i, j]:
                label = f"{share[i, j] * 100:.0f}" if share[i, j] >= 0.005 else "<1"
                ax.text(j, i, label, ha="center", va="center", fontsize=11,
                        color="#ffffff" if share[i, j] > 0.55 else INK2)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=12, color=INK)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t}  ({int(counts[i].sum()):,})" for i, t in enumerate(types)], fontsize=12, color=INK)
    ax.set_xlabel("gca's first suggestion", fontsize=13, color=INK2)
    ax.set_ylabel("type the author chose (commits)", fontsize=13, color=INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.04, 0.95, "Where the first suggestion goes, % of each row", fontsize=19, color=INK, weight="bold")
    fig.text(0.04, 0.915, f"{res['commits']:,} commits by people in 18 projects never used in training",
             fontsize=13, color=INK2)
    plt.subplots_adjust(left=0.2, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    results = json.loads((ROOT / "eval/results.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    out = ROOT / "docs/heldout_accuracy.png"
    draw(chart_text(results, split), out)
    print(f"wrote {out.relative_to(ROOT)}")
    out = ROOT / "out/confusion_matrix.png"
    if draw_confusion(results, out):
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
