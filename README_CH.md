# gca

[English](./README.md)

Git commit analyzer。gca 讀取變更，建議 Conventional Commit 類型，並讓你輸入
提交摘要。預設會接著 commit 與 push。分類模型內建於執行檔，在本機運算。

支援的類型：`feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、
`build`、`ci`、`chore`、`revert`。

## 安裝

Windows 使用者可執行 `gca-installer.exe`。安裝程式會將 `gca.exe` 複製到
`%LOCALAPPDATA%\gca`，並把該目錄加入使用者 PATH。如果找不到 Git，會嘗試
透過 `winget` 安裝。安裝後請開啟新的終端機。

若要自行編譯，先安裝 Git 與 Rust 工具鏈，再執行：

```sh
cd gca-rs
cargo build --release --bin gca
```

Windows 執行檔位於 `gca-rs/target/release/gca.exe`，其他平台則是
`gca-rs/target/release/gca`。可將所在目錄加入 PATH，或使用完整路徑執行。
只有訓練與驗證模型時需要 Python。

若要建立 Windows 安裝程式，先編譯 `gca`，再於 `gca-rs` 目錄執行：

```sh
cargo build --release --features installer --bin gca-installer
```

## 使用

在 Git 儲存庫內執行 gca。請先設定 Git 提交者身分與遠端儲存庫，方式與一般
commit、push 相同。

| 指令 | 行為 |
| --- | --- |
| `gca` | 使用暫存區變更；若暫存區為空，就暫存全部變更。選擇類型、輸入摘要後 commit 並 push。 |
| `gca ./src tests/foo.py` | 只暫存並提交指定路徑，再 push。其他已暫存變更會保留在暫存區。 |
| `gca list ./src` | 預覽哪些檔案會加入暫存區，不修改暫存區。 |
| `gca --dry-run` | 顯示建議，不詢問輸入、不修改暫存區，也不 commit 或 push。 |
| `gca --no-push` | commit 後不 push。 |
| `gca --confirm-push` | push 前先確認。 |
| `gca --remote origin` | 當次執行使用指定的遠端儲存庫。 |
| `gca --topk 5` | 顯示五個建議，預設為三個。 |
| `gca --model other.json` | 使用匯出的 JSON 模型，取代內建模型。 |

請將選項放在路徑之前，例如 `gca --no-push ./src`。`gca list` 未指定路徑時
會預覽全部變更。`gca --dry-run` 採用一般執行的路徑選擇規則，暫存操作則在
獨立的臨時 index 中進行。

當所有選取的檔案都符合 `docs`、`test` 或 `ci` 的路徑規則，gca 會預選該
類型。你仍可改選其他建議。

### 持久設定

`gca config` 將設定寫入全域 Git config。儲存庫設定與命令列選項可覆寫這些
預設值。

| 指令 | 行為 |
| --- | --- |
| `gca config push ask` | 每次 push 前先確認。 |
| `gca config push never` | 只 commit，不 push。 |
| `gca config push auto` | 直接 push，這是預設值。 |
| `gca config push` | 顯示目前的 push 設定。 |
| `gca config remote upstream` | 預設使用 `upstream` 遠端儲存庫。 |
| `gca config remote` | 顯示目前的遠端設定。 |

未指定遠端時，gca 執行 `git push`，依照 Git 自身的設定處理。
`--no-push` 與 `--confirm-push` 不能同時使用。若 `gca.push` 設定無效，
提交流程會在暫存變更之前停止。

## 模型

分類器使用經過機率校準的 LinearSVC，以 Conventional Commits 訓練。
特徵包含 diff 文字、檔案路徑、副檔名、新增與刪除詞彙的相似度，以及變更
統計。文字輸入限於 diff 的前 20,000 個字元。

`out/model_v2.json` 的權重會在編譯時嵌入。安裝後只需 Git，不需要 Python
環境或額外模型檔。匯出新權重後，請重新編譯 CLI，或透過 `--model` 指定
新的 JSON 檔。

### 重新訓練

請使用 Python 3.11 以上版本，從專案根目錄執行以下指令。先在虛擬環境安裝
已固定版本的依賴：

```sh
python -m pip install -r requirements.txt
```

從本機儲存庫收集訓練資料：

```sh
python miner.py --repo /path/to/repository --out datasets/local.jsonl
```

若要補充外部資料，先安裝 `datasets`，再匯入支援的資料來源。這個步驟需要
網路連線。

```sh
python -m pip install -r requirements-import.txt
python import_external.py --source commitbench --out datasets/commitbench.jsonl
```

合併資料後，依序訓練、匯出並驗證模型：

```sh
python dedupe.py --input datasets --output datasets/_merged.jsonl
python train_enhanced.py --data datasets/_merged.jsonl --model out/model_v2.joblib
python export_model.py
python verify_export.py
python gca-rs/gen_fixtures.py --synthetic --out gca-rs/tests/synthetic_fixtures.json
python gca-rs/gen_fixtures.py
cd gca-rs
cargo test --release
cargo build --release --bin gca
```

資料擷取程式會將結果附加到輸出檔。去重時會保留最先讀到的提交或正規化
diff，並排除輸出檔本身。若要優先採用某個來源，請在 `--input` 中將該檔案
排在前面。

專案附有合成測試資料，因此沒有訓練資料集也能執行 `cargo test`。重新訓練後
請重新產生兩份測試資料，讓預期機率與新模型一致。

`verify_export.py` 會先確認 JSON 參數與已儲存的 joblib 模型一致，再比較
預測結果。若參數不同，請先重新匯出已儲存的模型，再執行驗證。

### 模型評估

訓練程式會依類別比例保留約 10% 的資料作為測試集，資料量少時會提高比例，
並輸出評估指標與下方的混淆矩陣。匯出驗證與 Rust 一致性測試會比較兩端的
預測機率，不代表模型對新儲存庫的分類準確率。

![混淆矩陣](out/confusion_matrix.png)

diff 不一定能呈現修改意圖，尤其是 `refactor`、`fix` 與 `feat`。提交前仍應
確認建議類型是否符合這次變更。

### 儲存庫與時間評估

將來源儲存庫的本機 clone 放在 `temp_repos`，各儲存庫需設定 `origin`，
再從專案根目錄執行：

```sh
python evaluate_splits.py --data datasets/_merged.jsonl --repos-dir temp_repos --out out/evaluation
```

評估程式會移除 diff 前的提交資訊，以 SHA 與正規化 diff 去重，並排除標籤
衝突資料。預設比較 80/20 隨機切分與五折儲存庫切分，後者的訓練與測試
儲存庫互不重疊。對可透過 SHA 從本機 Git 歷史取得日期的資料，另以相同
母體比較約 80/20 的隨機與時間切分。日期採用 Git committer timestamp，
不使用舊的 `labeled_at` 欄位。

每個切分都重新訓練模型，不覆寫 `out/model_v2.joblib` 或匯出的模型。
結果與限制會寫入[評估報告](out/evaluation/report.md)。上述清理僅用於評估
程式，不會改變前面的訓練指令。

## 維護檢查

安裝 Python 依賴後，從專案根目錄執行：

```sh
python -m unittest discover -s tests -v
cd gca-rs
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```
