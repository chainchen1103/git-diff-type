# gca — Git Commit Analyzer

[English](./README.md)

一個指令從 dirty working tree 到 push 完成。
用 ML 分析 staged diff 建議 Conventional Commit 類型
（`feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`），
輸入 subject、commit、push 全在一次互動裡完成。

```
$ gca
no staged changes; running `git add -A`
Stats: +42 / -7 lines in 3 files

? Commit type
> feat      ( 71.3%)
  refactor  ( 14.9%)
  chore     (  6.2%)

? feat: add user login middleware
[main a1b2c3d] feat: add user login middleware
 3 files changed, 42 insertions(+), 7 deletions(-)
```

## 安裝

執行 `gca-installer.exe` — 會將 `gca.exe` 安裝到
`%LOCALAPPDATA%\gca`、加入使用者 PATH，並檢查 git 是否已安裝
（沒有的話會透過 `winget` 自動安裝）。

或自行編譯：

```
cd gca-rs
cargo build --release --bin gca
# 產出：target/release/gca.exe
```

## 使用

```
gca                        # 自動 stage 全部 → 選 type → commit → push
gca ./src tests/foo.py     # 只 stage 指定路徑 → commit → push
gca --dry-run              # 印出建議但不 commit
gca --no-push              # 只 commit 不 push
gca --confirm-push         # push 前確認
gca --remote origin        # 當次 push 到指定 remote
gca --model other.json     # 使用其他模型檔
```

### 持久設定

設定透過 `git config` 儲存，CLI flag 可覆寫單次行為。

```
gca config push ask         # 每次 push 前都問
gca config push never       # 只 commit 不 push
gca config push auto        # 自動 push（預設）
gca config push             # 印出目前設定

gca config remote upstream  # 永遠 push 到 upstream
gca config remote           # 印出目前 remote
```

### 啟發式

路徑啟發式涵蓋 `docs`、`test`、`ci` — 當所有 staged 檔案都符合同一條
規則時，會自動預選對應的 type，你仍然可以改選其他。

## 模型

分類器是用 Conventional Commit 的 repo 訓練的 calibrated LinearSVC。
模型權重在編譯時嵌入 binary，執行檔完全獨立（約 11 MB）。

### 重新訓練

```
python miner.py --repo <path> --out datasets/<name>.jsonl
python import_external.py --dataset commitbench --out datasets/commitbench.jsonl
python dedupe.py --input datasets/*.jsonl --out datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py
cd gca-rs && cargo build --release --bin gca
```

### 模型表現

![confusion_matrix](out/confusion_matrix.png)

`refactor` 類別表現最弱——語意主要在改動意圖，而非 diff 表面。
