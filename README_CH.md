# git-diff-type

[English](./README.md)

分析 staged git diff，判斷對應的 Conventional Commit 類型
(`feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`)，
並印出可以直接複製的 commit 指令。

```
$ git add .
$ gca
Stats: +42 / -7 lines in 3 files

? Pick commit type
> feat      ( 71.3%)
  refactor  ( 14.9%)
  chore     (  6.2%)

git commit -m "feat: <description>"
```

## 安裝

到 releases 下載 `gca.exe`，或自己編譯：

```
cd gca-rs
cargo build --release
# 產出：gca-rs/target/release/gca（Windows 是 gca.exe）
```

模型權重透過 `include_bytes!` 編入 binary，單一執行檔無需外部依賴。

## 使用

```
gca                     # 互動選 top-3（預設看 staged）
gca --dry-run           # 只印最高分的建議
gca --unstaged          # 分析未 staged 的改動
gca --model other.json  # 用外部 JSON 覆寫內建模型
```

會先套路徑啟發式（docs / test / ci），混合型 commit 才交給 ML 模型。

## 重新訓練

```
python miner.py --repo <path> --out datasets/<name>.jsonl
python import_external.py --dataset commitbench --out datasets/commitbench.jsonl
python dedupe.py --input datasets/*.jsonl --out datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py --model out/model_v2.joblib --out out/model_v2.json
cd gca-rs && cargo build --release
```

`export_model.py` 把 sklearn pipeline（TF-IDF vocab + IDF、CountVectorizer
vocabs、StandardScaler、LinearSVC 每個 CV fold 的 coef/intercept、sigmoid
校正）全部序列化成 JSON。`verify_export.py` 用純 NumPy 重實作 forward
pass，並和 `predict_proba` 比對誤差 <1e-6。

## 模型表現

![confusion_matrix](out/confusion_matrix.png)

`refactor` 類別表現最弱——這個類型的語意主要在改動意圖，而非 diff 表面。
