# gca

[English](./README.md)

**寫 commit message 時最卡的，往往是第一個字：這次到底算 `feat`、`fix` 還是 `refactor`？**

gca 讀取你準備 commit 的變更，用內建的 ML 模型排出最可能的 Conventional
Commit 類型，並從專案歷史建議 scope。你確認、輸入一行摘要，它就交給
`git commit` 完成。

- **離線、不需要 API key**：模型編譯進約 11 MB 的單一執行檔，diff 不會離開你的電腦。
- **在沒看過的專案上實測**：10,005 個來自 18 個從未用於訓練的專案、在訓練資料截止後才出現的
  commit，正確類型出現在它列出的三個選項中的比例是 **85.9%**。在其中舊版也沒看過的
  8 個專案上，第一個建議的正確率是 45.9%，舊版是 33.6%。
- **照 git 的規矩來**：不會自動暫存、不會自動 push；hooks、簽署、編輯器照常運作。

以下是 [shadcn/ui](https://github.com/shadcn-ui/ui) 的一個真實 commit（模型從沒看過這個專案），用 gca 重新 commit 的過程：

```
$ git add -A
$ gca
3 files  +188 -1
  A .changeset/fix-registry-header-redirect-leak.md
  A packages/shadcn/src/registry/proxy.test.ts
  M packages/shadcn/src/registry/proxy.ts
✔ Commit type · fix       ( 21.5%)
✔ Scope (optional) · shadcn
✔ fix(shadcn): · drop custom registry headers on cross-origin redirects
[fix-redirect-headers fc3ada1d] fix(shadcn): drop custom registry headers on cross-origin redirects
 3 files changed, 188 insertions(+), 1 deletion(-)
 create mode 100644 .changeset/fix-registry-header-redirect-leak.md
 create mode 100644 packages/shadcn/src/registry/proxy.test.ts
```

支援的類型：`feat`、`fix`、`docs`、`style`、`refactor`、`perf`、`test`、
`build`、`ci`、`chore`、`revert`。

## 安裝

**macOS / Linux**：

```
curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh
```

**Windows**（PowerShell）：

```
irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex
```

腳本會從最新的 [Release](https://github.com/chainchen1103/git-diff-type/releases)
下載你的平台對應的 gca，用 Release 附的 `SHA256SUMS` 驗證，放到 `~/.local/bin`
（Windows：`%LOCALAPPDATA%\gca`）並把該資料夾加入 PATH，其他東西都不會動。再執行一次就是升級。
移除：`curl -fsSL .../install.sh | sh -s -- --uninstall`；PowerShell 則先設定
`$env:GCA_UNINSTALL = 1` 再執行同一行。`GCA_VERSION=v0.2.0` 可指定版本，`GCA_INSTALL_DIR`
指定資料夾，`GCA_NO_MODIFY_PATH=1` 則不修改 PATH。

預先編譯的執行檔涵蓋 Windows x64（在 Windows on Arm 上也能執行）、macOS（Apple silicon 與 Intel）
和 Linux x86_64。gca 需要 git。

其他安裝方式：

- **Windows 安裝程式**：Releases 裡的 `gca-installer.exe` 做的事和 PowerShell 腳本相同，
  找不到 git 時還會詢問是否用 `winget` 安裝。`gca-installer --uninstall` 可移除。
- **用 Rust 1.80 以上編譯**（任何平台）：
  `cargo install --git https://github.com/chainchen1103/git-diff-type gca-rs --bin gca`
- **從原始碼編譯**：`cd gca-rs && cargo build --release --bin gca` 會產生
  `target/release/gca`（Windows 為 `gca.exe`）。安裝程式會嵌入這個檔案，所以要第二步再編：
  `cargo build --release --bin gca-installer --features installer`。

Shell 補全：`gca completions bash|zsh|fish|powershell`，例如 `gca completions zsh > ~/.zfunc/_gca`。

## 使用

gca 的行為和 `git commit` 一致：它 commit **已暫存（staged）的內容**。

```
git add -p && gca          # 先挑好要 commit 的內容，再執行 gca
gca -a                     # commit 所有已追蹤檔案的變更，同 git commit -a（不含新檔案）
gca src/auth tests/auth    # 只 commit 這些路徑（包含新檔案），其他已暫存的內容保持不動
```

沒有任何暫存的變更時，gca 不會替你暫存，而是列出 `git status` 和做法後結束。

互動流程：

1. **類型**：依機率排序，按 Enter 採用預選的；不在前幾名時選「other type…」。
2. **Scope**：只有專案本身在用 scope 時才會問。預填的是**這些檔案過去最常用的 scope**
   （從最近 500 個 commit 學來，不計 bot），可直接改或清空。
3. **摘要**：一行描述；整行 header 超過 100 字元會被擋下（commitlint 的預設上限）。

Esc 或 Ctrl-C 隨時取消，暫存區不受影響。`-a` 和指定路徑都先在暫存區的臨時副本上試算，
取消時真正的暫存區完全不變。

### 選項

```
-t, --type <TYPE>     指定類型，不詢問
    --scope <SCOPE>   指定 scope，不詢問；"" 代表不加
-m, --message <MSG>   摘要；重複使用可加上內文段落，同 git commit -m
-b, --breaking        標記為 breaking change（feat!: ...）
-y, --yes             直接採用建議的類型與 scope
-e, --edit            commit 前用編輯器開啟訊息
-n, --no-verify       略過 pre-commit 與 commit-msg hook
-s, --signoff         加上 Signed-off-by
    --push            commit 後 push；沒有 upstream 時自動設定
    --confirm-push    push 前先確認；--no-push 則不 push
    --remote <NAME>   push 到這個遠端，而不是分支的 upstream
    --dry-run         只顯示建議，不暫存、不 commit
    --dry-run --json  輸出 JSON，給腳本或編輯器整合使用
    --topk <N>        列出幾個建議（預設 3）
    --model <FILE>    使用匯出的 JSON 模型，取代內建模型
```

不需互動、可放進腳本：`gca -a -t fix -m "handle empty diff"`（專案有用 scope 時再加上
`--scope <scope>`，不要 scope 則用 `--scope ""`），或 `gca -y -m "update guide"` 直接採用建議的類型與 scope。
沒有終端機時，gca 會列出還缺哪些參數，而不是卡住。

結束代碼：`0` 已 commit（或 dry run 完成）、`1` 沒有可 commit 的內容／取消／git 失敗、
`2` 參數錯誤、`130` Ctrl-C。

### 設定

設定存在 git config；預設寫入全域，加 `--local` 只作用於目前的儲存庫。命令列選項優先。

```
gca config push              # 顯示目前設定（預設 never：不 push）
gca config push ask          # commit 後詢問是否 push
gca config push auto --local # 這個儲存庫每次都 push
gca config remote upstream   # push 到這裡（未設定時：分支的 upstream，其次 origin）
```

`gca.push` 的值無效時，gca 會在暫存任何東西之前就停止。

### 預選規則

當所有變更的檔案都屬於文件、測試或 CI 設定（例如 `docs/`、`*_test.go`、`.github/workflows/`），
會預選 `docs`、`test` 或 `ci`（前提是模型有這個類型），你仍可改選其他類型。

## 準確率

模型以 **437,945 個 commit** 訓練，來自 5,925 個儲存庫：其中 342,347 個是 36 個遵循 Conventional Commits 的開源專案裡由人寫、2026-04-20 以前的 commit（不含 bot），另外 95,598 個來自公開資料集 CommitChronicle 與 CommitBench，並已排除所有測試專案所屬組織的資料。測試資料取自公開儲存庫，分成三組：

| 測試資料 | commit 數 | 第一個建議正確 | 正確類型在前 3 | 各類型平均召回率 | 對照：永遠猜最常見類型 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **沒看過的專案**，訓練後才出現的 commit | 10,005 | **43.5%** | **85.9%** | 42.9% | 44.1% (`fix`) |
| 沒看過的專案，較早的歷史 | 91,617 | 44.6% | 83.7% | 36.2% | 35.2% (`fix`) |
| 訓練過的專案，訓練後才出現的 commit | 39,591 | 58.0% | 89.7% | 44.9% | 39.2% (`fix`) |

「沒看過的專案」共 18 個，例如 vite、vue、electron、nest、superset、rolldown，與所有訓練資料分屬
不同組織；任何與訓練資料 diff 相同的 commit 都已移除。以上都只計算由人寫的 commit；
bot 產生的 commit 另外統計在 `eval/results.json`。

**各類型平均召回率**是每個類型中第一個建議就正確的比例，再對類型取平均。在第一列，永遠回答 `fix`
的第一個建議會有 44.1% 是對的，但它的各類型平均召回率只有 11.1%，因為其他類型
一個都不會對。gca 的第一個建議則分散到各類型，其餘交給三個選項的清單。

![各類型的實測準確率](docs/heldout_accuracy.png)

與舊模型在同一批 commit 上的比較（舊 → 新）：

| 測試資料 | 第一個建議正確 | 正確類型在前 3 | 各類型平均召回率 |
| --- | ---: | ---: | ---: |
| 沒看過的專案，訓練後的 commit | 42.0% → 43.5% | 75.1% → 85.9% | 44.1% → 42.9% |
| … 其中舊模型也沒看過的 8 個專案 | 33.6% → 45.9% | 68.1% → 86.8% | 42.6% → 47.4% |
| 沒看過的專案，較早的歷史 | 41.6% → 44.6% | 73.3% → 83.7% | 41.7% → 36.2% |
| … 其中舊模型也沒看過的 8 個專案 | 32.2% → 46.4% | 65.7% → 84.8% | 37.4% → 34.6% |
| 訓練過的專案，訓練後的 commit | 33.5% → 58.0% | 65.2% → 89.7% | 37.5% → 44.9% |

舊模型的訓練資料含有 18 個測試專案中 10 個專案的 commit（例如 quasar、electron、vue core、nest），這些專案對它來說並非沒看過；「較早的歷史」更可能有部分就是它訓練過的 commit。兩個模型都沒看過的 8 個專案（hoppscotch, napi-rs, novu, rolldown, shadcn/ui, vitest, vueuse, zitadel）才是公平的比較：在那裡，新模型的第一個建議與前三個選項都明顯較好。舊模型表現最好的是各類型平均召回率，尤其在較早的歷史上：它更常把少見的類型排第一，代價是整體猜錯得更多。

`docs`、`test`、`ci` 最準，因為檔案路徑本身就有訊號。`refactor` 與 `perf` 最弱：它們和其他
變更的差別在於**意圖**，diff 很少透露。這也是 gca 給你排序好的清單、而不是替你決定的原因；
commit 前請確認建議的類型。

完整數字與重現方式見 [eval/README.md](eval/README.md)。

## 運作方式

![架構](docs/architecture.png)

1. **資料**：`miner.py` 從遵循 Conventional Commits 的儲存庫挖出 (diff, type) 配對；
   `split_dataset.py` 依專案與日期切出訓練集和三組測試集。`import_external.py` 與
   `eval/prepare_external.py` 加入公開資料集，同時避開測試專案。
2. **訓練**：TF-IDF（diff 內容）＋檔案路徑與副檔名詞袋＋新增與刪除詞彙的重疊程度＋增刪行數，
   餵給機率校準過的 LinearSVC，輸出 11 個類型的機率。diff 只讀前 20,000 個字元。
   接著依各類型在訓練資料中的常見程度修正機率（`eval/tune_prior.py`）；不這樣做的話，
   大部分的 diff 都會被判成 `fix`，`refactor`、`perf` 幾乎不會排第一。
3. **匯出**：`export_model.py` 把整條 sklearn pipeline 攤平成 `out/model_v2.json`。
4. **推論**：Rust CLI 在編譯時嵌入這份 JSON，自己實作同樣的前向運算，所以安裝後只需要 git，
   不需要 Python 或模型檔。`gca-rs/tests/parity.rs` 驗證 Rust 與 Python 的機率誤差小於 1e-6。
   匯出新權重後，重新編譯 CLI，或用 `--model` 指定 JSON。

gca 讀 diff 時固定使用 git 的預設格式（不受 `diff.noprefix`、`color.ui` 等個人設定影響），
和訓練資料的格式一致。

### 重新訓練

需要 Python 3.11+ 與 `pip install -r requirements.txt`，在專案根目錄執行：

```
bash eval/collect.sh                    # 下載 eval/repos_*.txt 的專案、挖掘、切分到 datasets/
pip install -r requirements-import.txt  # 公開資料集（需要網路）
python import_external.py --source commitchronicle --out datasets/commitchronicle.jsonl
python import_external.py --source commitbench --out datasets/commitbench.jsonl
python eval/prepare_external.py --inputs datasets/commitchronicle.jsonl datasets/commitbench.jsonl \
    --test-repos eval/repos_test.txt --out datasets/external.jsonl \
    --against datasets/train.jsonl datasets/test_*.jsonl
python eval/tune_c.py --data datasets/train.jsonl   # 只用訓練專案挑選正則化強度 C
python train_enhanced.py --stream --data datasets/train.jsonl datasets/external.jsonl --C 0.1
python eval/tune_prior.py --write 0.9   # 依類型常見程度修正（見 eval/README.md）
python export_model.py
python verify_export.py --data datasets/test_unseen_recent.jsonl
python gca-rs/gen_fixtures.py --synthetic --out gca-rs/tests/synthetic_fixtures.json
python gca-rs/gen_fixtures.py --data datasets/test_unseen_recent.jsonl
bash eval/run.sh                        # 在三組測試集上評估
cd gca-rs && cargo test --release && cargo build --release --bin gca
```

`verify_export.py` 會先確認 JSON 參數與已儲存的 joblib 模型一致，再比較預測結果；
`--check-fixtures` 則改為拿 JSON 和一致性測試資料比對。專案附有合成測試資料，沒有資料集也能跑
`cargo test`；由真實 diff 產生的 `gca-rs/tests/fixtures.json` 只留在本機。重新訓練後兩份都要重新產生。

若想改用自己的歷史訓練：用 `python miner.py --repo <path> --out datasets/local.jsonl` 挖掘
（結果會附加在檔尾），用 `python dedupe.py --input datasets --output datasets/_merged.jsonl`
合併（保留最先讀到的 commit 或正規化 diff，想優先採用的來源請排在前面），再不加 `--stream`
訓練合併後的檔案；這種模式也會保留 10% 分層測試集並輸出 `out/confusion_matrix.png`。

### 其他評估

`evaluate_splits.py` 會重新訓練模型，比較隨機 80/20 切分與訓練、測試儲存庫互不重疊的五折切分；
對能從本機 clone 讀到日期的 commit，再比較隨機切分與依時間切分：

```
python evaluate_splits.py --data datasets/_merged.jsonl --repos-dir temp_repos --out out/evaluation
```

它會移除 diff 中的 commit 標頭、依 SHA 與正規化 diff 去重、排除標籤衝突的資料，不會改動發佈的模型。

## 開發檢查

```
python -m unittest discover -s tests -v
cd gca-rs
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release
```

CI 在每次 push 時執行這些檢查（Rust 部分在 Windows、macOS、Linux 上各跑一次），並在 Windows 上編譯安裝程式。

## Roadmap

- 依使用者在自己儲存庫的選擇做本地微調
- 讀取專案的 commitlint 設定（自訂類型、header 長度）
- 改善 `refactor` / `perf`：加入「行為是否改變」相關的特徵

[更新紀錄](CHANGELOG.md)
