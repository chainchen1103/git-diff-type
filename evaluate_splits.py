"""Evaluate fresh classifiers on random, repository, and chronological splits."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import time
import warnings
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from dedupe import normalize_diff
from import_external import ALL_LABELS
from train_enhanced import FEATURE_COLUMNS, build_model, prepare_features

LABELS = sorted(ALL_LABELS)
PATCH_START = re.compile(r"^diff --git ", re.MULTILINE)
SHA = re.compile(r"^[0-9a-f]{40,64}$")


def utc_date(timestamp):
    return datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat().replace("+00:00", "Z")


def remote_identity(remote):
    remote = remote.strip()
    if "://" in remote:
        path = urlsplit(remote).path
    elif re.match(r"[^/]+@[^:]+:", remote):
        path = remote.split(":", 1)[1]
    else:
        return None
    parts = path.strip("/").removesuffix(".git").split("/")
    return "/".join(parts).lower() if len(parts) == 2 else None


def git_metadata(repos_dir):
    aliases, commit_times = {}, {}
    for repo in sorted(Path(repos_dir).iterdir()):
        if not (repo / ".git").exists():
            continue
        origin = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                                capture_output=True, text=True, encoding="utf-8", check=True)
        identity = remote_identity(origin.stdout)
        if identity is None:
            raise ValueError(f"cannot identify repository {repo.name}")
        aliases[repo.name.lower()] = identity
        log = subprocess.run(["git", "-C", str(repo), "log", "--all", "--format=%H %ct"],
                             capture_output=True, text=True, encoding="utf-8", check=True)
        for line in log.stdout.splitlines():
            sha, timestamp = line.split()
            commit_times[sha] = int(timestamp)
    return aliases, commit_times


def repository_identity(row, aliases):
    source = str(row.get("url", "")).partition("://")[0]
    owner = str(row.get("owner") or "").strip().lower()
    repo = str(row.get("repo") or "").strip().lower().removesuffix(".git")
    if source == "local" or owner == "local":
        return aliases.get(repo)
    if source == "cb" and owner == repo and "_" in repo:
        owner, repo = repo.split("_", 1)
    if "/" in repo:
        parts = repo.split("/")
        if len(parts) != 2:
            return None
        owner, repo = parts
    if not owner or not repo or owner == "unknown" or repo == "unknown":
        return None
    return f"{owner}/{repo}"


def clean_diff(text, max_diff_len=20000):
    if not isinstance(text, str):
        return None
    match = PATCH_START.search(text)
    if match is None:
        return None
    return text[match.start():].strip()[:max_diff_len]


def deduplicate(frame):
    parents = list(range(len(frame)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for column in ("sha", "diff_hash"):
        seen = {}
        for index, value in enumerate(frame[column]):
            if not value:
                continue
            if value in seen:
                parents[find(index)] = find(seen[value])
            else:
                seen[value] = index
    groups = defaultdict(list)
    for index in range(len(frame)):
        groups[find(index)].append(index)
    keep = []
    dropped_duplicates = dropped_conflicts = conflict_groups = 0
    labels = frame.label.to_numpy()
    times = frame.commit_time.to_numpy()
    for indices in groups.values():
        if len({labels[index] for index in indices}) > 1:
            conflict_groups += 1
            dropped_conflicts += len(indices)
            continue
        first = min(indices, key=lambda index: (times[index] if pd.notna(times[index]) else float("inf"), index))
        keep.append(first)
        dropped_duplicates += len(indices) - 1
    return frame.iloc[sorted(keep)].reset_index(drop=True), {
        "duplicate_rows_removed": dropped_duplicates,
        "conflicting_label_groups_removed": conflict_groups,
        "conflicting_label_rows_removed": dropped_conflicts,
    }


def load_corpus(data_path, repos_dir):
    aliases, commit_times = git_metadata(repos_dir)
    audit = Counter()
    sources = Counter()
    rows = []
    digest = hashlib.sha256()
    with Path(data_path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            audit["input_rows"] += 1
            row = json.loads(raw)
            if not isinstance(row, dict) or row.get("label") not in LABELS:
                audit["invalid_label_rows"] += 1
                continue
            diff = clean_diff(row.get("diff_text"))
            if not diff:
                audit["missing_patch_rows"] += 1
                continue
            if PATCH_START.search(row["diff_text"]).start() > 0:
                audit["preamble_removed_rows"] += 1
            repo = repository_identity(row, aliases)
            if repo is None:
                audit["unknown_repository_rows"] += 1
                continue
            source = str(row.get("url", "")).partition("://")[0]
            sources[source] += 1
            sha = str(row.get("sha") or "").lower()
            sha = sha if SHA.fullmatch(sha) else ""
            normalized = normalize_diff(diff)
            if not normalized:
                audit["empty_patch_rows"] += 1
                continue
            rows.append({
                "row_id": line_number, "diff_text": diff, "label": row["label"],
                "repository": repo, "source": source, "sha": sha,
                "diff_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                "commit_time": commit_times.get(sha),
                "files_changed": row.get("files_changed", 0),
                "additions": row.get("additions", 0), "deletions": row.get("deletions", 0),
            })
    if not rows:
        raise ValueError("no usable patches with repository identities")
    frame, dedupe_audit = deduplicate(pd.DataFrame(rows))
    frame = prepare_features(frame)
    audit.update(dedupe_audit)
    dates = frame.commit_time.dropna()
    metadata = {
        **dict(audit), "data_sha256": digest.hexdigest(), "retained_rows": len(frame),
        "retained_repositories": int(frame.repository.nunique()), "local_repository_aliases": aliases,
        "source_rows_before_deduplication": dict(sources),
        "source_rows": frame.source.value_counts().to_dict(), "label_counts": frame.label.value_counts().to_dict(),
        "trusted_time_rows": len(dates), "rows_without_trusted_time": int(frame.commit_time.isna().sum()),
        "trusted_time_repositories": int(frame.loc[frame.commit_time.notna(), "repository"].nunique()),
        "timestamp_source": "Git committer Unix timestamp, recovered by SHA from local repositories",
        "timestamp_range": [utc_date(dates.min()), utc_date(dates.max())] if len(dates) else None,
    }
    return frame, metadata


def chronological_split(frame, test_size):
    if len(frame) < 2:
        raise ValueError("chronological split needs at least two rows")
    ordered = frame.sort_values(["commit_time", "row_id"])
    if ordered.commit_time.isna().any():
        raise ValueError("chronological split requires trusted commit timestamps")
    position = min(len(ordered) - 1, max(1, int(len(ordered) * (1 - test_size))))
    cutoff = int(ordered.iloc[position].commit_time)
    train = ordered.index[ordered.commit_time < cutoff].to_numpy()
    test = ordered.index[ordered.commit_time >= cutoff].to_numpy()
    if not len(train) or not len(test):
        raise ValueError("timestamps do not allow a nonempty chronological split")
    return train, test, cutoff


def validate_split(train, test, repository_split=False, temporal_split=False):
    if train.empty or test.empty:
        raise ValueError("train and test must both contain rows")
    for column in ("row_id", "sha", "diff_hash"):
        shared = set(train[column]) & set(test[column]) - {""}
        if shared:
            raise ValueError(f"train/test overlap in {column}: {len(shared)}")
    if repository_split and set(train.repository) & set(test.repository):
        raise ValueError("repository split shares repositories between train and test")
    if temporal_split:
        if train.commit_time.isna().any() or test.commit_time.isna().any():
            raise ValueError("temporal split contains missing timestamps")
        if train.commit_time.max() >= test.commit_time.min():
            raise ValueError("temporal split contains overlapping or reversed timestamps")
    counts = train.label.value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(f"calibration needs at least three training rows per label: {counts.to_dict()}")


def metrics(y_true, predictions, top3):
    report = classification_report(y_true, predictions, labels=LABELS, output_dict=True, zero_division=0)
    represented = sorted(set(y_true))
    return {
        "rows": len(y_true), "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=LABELS, average="weighted", zero_division=0)),
        "balanced_accuracy": float(np.mean([report[label]["recall"] for label in represented])),
        "top3_accuracy": float(np.mean(top3)), "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=LABELS).tolist(),
    }


def evaluate_fold(frame, train_indices, test_indices, name, output_dir, seed, temporal=False, grouped=False, cutoff=None):
    train, test = frame.loc[train_indices], frame.loc[test_indices]
    validate_split(train, test, repository_split=grouped, temporal_split=temporal)
    print(f"{name}: fitting {len(train):,} rows; testing {len(test):,} rows", flush=True)
    started = time.perf_counter()
    model = build_model()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[FEATURE_COLUMNS], train.label)
        probabilities = model.predict_proba(test[FEATURE_COLUMNS])
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name}: non-finite predictions")
    classes = model.classes_
    predictions = classes[probabilities.argmax(axis=1)]
    choices = classes[np.argsort(-probabilities, axis=1, kind="stable")[:, :3]]
    correct_top3 = np.any(choices == test.label.to_numpy()[:, None], axis=1)
    result = metrics(test.label.to_numpy(), predictions, correct_top3)
    majority = train.label.value_counts().index[0]
    majority_predictions = np.repeat(majority, len(test))
    result.update({
        "name": name, "seed": seed, "train_rows": len(train), "test_rows": len(test),
        "train_repositories": int(train.repository.nunique()), "test_repositories": int(test.repository.nunique()),
        "repository_overlap": len(set(train.repository) & set(test.repository)),
        "train_label_counts": train.label.value_counts().to_dict(), "test_label_counts": test.label.value_counts().to_dict(),
        "unseen_test_labels": sorted(set(test.label) - set(train.label)),
        "majority_class": majority, "majority_accuracy": float(accuracy_score(test.label, majority_predictions)),
        "majority_macro_f1": float(f1_score(test.label, majority_predictions, labels=LABELS, average="macro", zero_division=0)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "warnings": sorted({str(item.message) for item in caught}),
        "cutoff_utc": utc_date(cutoff) if cutoff is not None else None,
        "train_time_range": [utc_date(train.commit_time.min()), utc_date(train.commit_time.max())] if temporal else None,
        "test_time_range": [utc_date(test.commit_time.min()), utc_date(test.commit_time.max())] if temporal else None,
        "train_row_ids_sha256": hashlib.sha256(train.row_id.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
        "test_row_ids_sha256": hashlib.sha256(test.row_id.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
    })
    prediction_frame = pd.DataFrame({
        "row_id": test.row_id.to_numpy(), "repository": test.repository.to_numpy(),
        "label": test.label.to_numpy(), "prediction": predictions, "top3_correct": correct_top3,
        "majority_prediction": majority, "fold": name,
    })
    prediction_frame.to_csv(output_dir / f"{name}_predictions.csv.gz", index=False, compression="gzip")
    (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{name}: accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f} top3={result['top3_accuracy']:.4f}; {result['elapsed_seconds']:.1f}s", flush=True)
    return result, prediction_frame


def pooled_repositories(predictions):
    output = predictions.sort_values("row_id").reset_index(drop=True)
    result = metrics(output.label.to_numpy(), output.prediction.to_numpy(), output.top3_correct.to_numpy())
    result["majority_accuracy"] = float(accuracy_score(output.label, output.majority_prediction))
    result["majority_macro_f1"] = float(f1_score(output.label, output.majority_prediction, labels=LABELS, average="macro", zero_division=0))
    per_repo = []
    for repo, rows in output.groupby("repository", sort=True):
        per_repo.append({
            "repository": repo, "rows": len(rows), "accuracy": float(accuracy_score(rows.label, rows.prediction)),
            "macro_f1_present_labels": float(f1_score(rows.label, rows.prediction, labels=sorted(set(rows.label)), average="macro", zero_division=0)),
        })
    result["repository_equal_weight_accuracy"] = float(np.mean([row["accuracy"] for row in per_repo]))
    result["repository_count"] = len(per_repo)
    result["per_repository"] = per_repo
    return result


def write_report(results, output_dir):
    summaries = results["summaries"]
    names = {"random": "全資料隨機切分", "repository": "跨儲存庫五折", "dated_random": "有可信日期資料的隨機切分", "chronological": "依提交時間切分"}
    audit = results["data"]
    lines = [
        "# 分類模型重新評估", "", "每個切分都重新訓練完整模型，特徵設定與現有 CLI 相同。現有正式模型未被載入或覆寫。", "",
        "| 評估方式 | 測試筆數 | Accuracy | Macro-F1 | Top-3 | 多數類基準 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in summaries.items():
        title = names[name].replace("五折", f"{results['config']['group_folds']} 折")
        lines.append(f"| {title} | {value['rows']:,} | {value['accuracy']:.2%} | {value['macro_f1']:.4f} | {value['top3_accuracy']:.2%} | {value['majority_accuracy']:.2%} |")
    lines.extend([
        "", f"資料共 {audit['input_rows']:,} 筆，清理後保留 {audit['retained_rows']:,} 筆、{audit['retained_repositories']:,} 個儲存庫。",
        f"移除 {audit.get('preamble_removed_rows', 0):,} 筆 diff 前的提交訊息與 metadata，避免提交類型洩漏。",
        f"另排除 {audit['duplicate_rows_removed']:,} 筆相同 SHA 或正規化 diff 副本，以及 {audit['conflicting_label_rows_removed']:,} 筆標籤衝突資料。",
        "", f"時間評估使用 {audit['trusted_time_rows']:,} 筆能由本機 Git SHA 驗證的 committer timestamp。",
        "原資料的 labeled_at 不用於切分：本機資料含時區標記錯誤，CommitBench 為匯入時間，CommitChronicle 缺少原始時區。",
        "有可信日期的隨機切分和時間切分使用相同的資料母體，但測試提交不同，差距同時反映時間與樣本組成變化。",
        "", "儲存庫切分的訓練與測試儲存庫沒有交集；所有切分都檢查 SHA 與正規化 diff 沒有交集。",
        "重複資料優先保留有可信日期的最早版本。時間切分以單一截止點分隔，同一秒的提交不跨側。",
        "Macro-F1 固定計算全部 11 類。Top-3 衡量前三個模型建議包含正確類型的比例，不含 CLI 路徑規則。",
        "", "## 各類別表現", "", "| 類型 | 隨機 F1 | 跨儲存庫 F1 | 日期母體隨機 F1 | 時間切分 F1 | 時間測試筆數 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in LABELS:
        values = [summaries[name]["per_class"][label] for name in names]
        lines.append(f"| {label} | " + " | ".join(f"{value['f1-score']:.4f}" for value in values) + f" | {int(values[-1]['support']):,} |")
    lines.extend(["", "## 跨儲存庫各折", "", "| 折 | 訓練筆數 | 測試筆數 | Accuracy | Macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"])
    for fold in results["folds"]:
        if fold["name"].startswith("repository_"):
            lines.append(f"| {fold['name']} | {fold['train_rows']:,} | {fold['test_rows']:,} | {fold['accuracy']:.2%} | {fold['macro_f1']:.4f} |")
    temporal = summaries["chronological"]
    lines.extend([
        "", f"時間截止點：`{temporal['cutoff_utc']}`。訓練資料早於此時刻，測試資料從此時刻開始。",
        "", "## 解讀限制", "",
        f"跨儲存庫分數採所有折的測試預測合併計算；每筆資料恰好測試一次。全資料隨機基準只測試其中 {results['config']['test_size']:.0%}，不能視為相同測試集的成對實驗。",
        "時間結果只代表能核實日期的儲存庫。大儲存庫與常見類別會影響整體 accuracy，應一起看 Macro-F1、各類支持筆數與各折差異。",
        "本次排除了相同 SHA 與正規化 diff，沒有完全排除不同 SHA 的近似修改或相關 fork，因此仍可能高估泛化能力。",
        "既有混淆矩陣來自不同資料處理與切分，不能用來宣稱本次模型更好或更差。未以測試結果調整參數。",
        "", "## 重現", "", "```sh", results["command"], "```", "",
        "完整指標、類別筆數、資料與程式雜湊保存在 `results.json`。`corpus_manifest.csv.gz` 保存每筆保留資料的來源、SHA、類別、儲存庫與可信時間；每折預測也保存在壓縮 CSV，供核對切分與統計。", "",
    ])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/_merged.jsonl")
    parser.add_argument("--repos-dir", default="temp_repos")
    parser.add_argument("--out", default="out/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--group-folds", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or args.group_folds < 2:
        parser.error("test-size must be between 0 and 1; group-folds must be at least two")
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Reading and cleaning corpus...", flush=True)
    frame, audit = load_corpus(args.data, args.repos_dir)
    manifest = output_dir / "corpus_manifest.csv.gz"
    frame[["row_id", "repository", "sha", "diff_hash", "commit_time", "label", "source"]].to_csv(
        manifest, index=False, compression={"method": "gzip", "mtime": 0})
    audit["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    print(json.dumps(audit, ensure_ascii=True, indent=2), flush=True)
    (output_dir / "corpus.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    argv = ["python", "evaluate_splits.py", "--data", args.data, "--repos-dir", args.repos_dir,
            "--out", args.out, "--seed", str(args.seed), "--test-size", str(args.test_size), "--group-folds", str(args.group_folds)]
    results = {
        "data": audit, "config": vars(args), "labels": LABELS,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("evaluate_splits.py", "train_enhanced.py", "dedupe.py")},
        "argv": argv, "command": subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv),
        "folds": [], "summaries": {},
    }

    def record(train, test, name, **options):
        value, predictions = evaluate_fold(frame, train, test, name, output_dir, args.seed, **options)
        results["folds"].append(value)
        (output_dir / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        return value, predictions

    train, test = train_test_split(frame.index.to_numpy(), test_size=args.test_size, random_state=args.seed, stratify=frame.label)
    results["summaries"]["random"], _ = record(train, test, "random")
    group_predictions = []
    splitter = GroupKFold(n_splits=args.group_folds)
    for fold, (train, test) in enumerate(splitter.split(frame, frame.label, frame.repository), 1):
        _, predictions = record(train, test, f"repository_{fold}", grouped=True)
        group_predictions.append(predictions)
    pooled = pd.concat(group_predictions, ignore_index=True)
    if pooled.row_id.nunique() != len(frame) or len(pooled) != len(frame):
        raise ValueError("repository folds must test every retained row exactly once")
    results["summaries"]["repository"] = pooled_repositories(pooled)
    dated = frame[frame.commit_time.notna()]
    train, test = train_test_split(dated.index.to_numpy(), test_size=args.test_size, random_state=args.seed, stratify=dated.label)
    results["summaries"]["dated_random"], _ = record(train, test, "dated_random")
    train, test, cutoff = chronological_split(dated, args.test_size)
    results["summaries"]["chronological"], _ = record(train, test, "chronological", temporal=True, cutoff=cutoff)
    results["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    write_report(results, output_dir)
    print(f"Finished in {results['elapsed_seconds']:.1f}s; report: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
