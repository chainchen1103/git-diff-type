# Git diff type  
[English](./README.md)  
用於分類 git 更變類型輕量化模型  
透過 Github 收集的 commit 進行訓練

## model performance
![confusion_matrix](out/confusion_matrix.png)  
## Usage 
git 環境中在 CLI 使用 `gca` 會根據模型推論更動類型 `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`
並產生對應的 commit message。
