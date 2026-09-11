# gca

[English](./README.md)

Git commit analyzer。一個指令從 dirty working tree 走到 push 完成：讀取
staged diff，用 ML 模型建議 Conventional Commit 類型，輸入 subject 後直接
commit 並 push。

支援的類型：`feat` `fix` `docs` `style` `refactor` `perf` `test`
`build` `ci` `chore` `revert`。

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

執行 `gca-installer.exe`，它會把 `gca.exe` 複製到 `%LOCALAPPDATA%\gca` 並
加入使用者 PATH。電腦上沒有 git 的話，會用 `winget` 裝好。

或自行編譯：

```
cd gca-rs
cargo build --release --bin gca
# 產出：target/release/gca.exe
```

installer 會把 `target/release/gca.exe` 包進去，所以要先編 `gca`，再執行
`cargo build --release --bin gca-installer`。

## 使用

```
gca                        # 沒有 staged 變更時自動 stage 全部，選 type、commit、push
gca ./src tests/foo.py     # 只 stage 指定路徑，再 commit、push
gca list ./src             # 列出會被 stage 的檔案，不實際 stage
gca --dry-run              # 只印建議，不 commit
gca --no-push              # 只 commit 不 push
gca --confirm-push         # push 前先問
gca --remote origin        # 這次 push 到指定 remote
gca --model other.json     # 改用其他模型檔
```

### 持久設定

設定存在全域 git config，CLI flag 只覆寫當次執行。

```
gca config push ask         # 每次 push 前都問
gca config push never       # 只 commit 不 push
gca config push auto        # 直接 push，預設值
gca config push             # 印出目前設定

gca config remote upstream  # 固定 push 到 upstream
gca config remote           # 印出目前 remote
```

### 啟發式

所有 staged 檔案都符合 `docs`、`test` 或 `ci` 的路徑規則時，會預選該類型，
你還是可以改選別的。

## 模型

分類器是 calibrated LinearSVC，訓練資料來自採用 Conventional Commits 的
repo。權重在編譯時嵌入，11 MB 的執行檔不需要其他檔案。

### 重新訓練

```
python miner.py --repo <path> --out datasets/<name>.jsonl
python import_external.py --source commitbench --out datasets/commitbench.jsonl
python dedupe.py --input datasets/*.jsonl --output datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py
python gca-rs/gen_fixtures.py
cd gca-rs && cargo test --release && cargo build --release --bin gca
```

### 模型表現

![confusion_matrix](out/confusion_matrix.png)

`refactor` 最弱。它的訊號多半藏在改動的意圖裡，diff 本身很難看出來。
