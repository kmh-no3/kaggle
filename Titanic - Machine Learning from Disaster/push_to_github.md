# 🚀 GitHubへのPush手順

## 📋 前提条件
- GitHubアカウントを持っていること
- Gitがローカルにインストールされていること

## 🔧 手順

### 1. GitHubでリポジトリを作成
1. [GitHub.com](https://github.com)にアクセス
2. 右上の「+」ボタンをクリック → 「New repository」を選択
3. リポジトリ名を入力（例：`titanic-ml-project`）
4. 説明を追加（例：`Titanic survival prediction using machine learning`）
5. **重要**: README、.gitignore、ライセンスは作成しない（既に存在するため）
6. 「Create repository」をクリック

### 2. リモートリポジトリを追加
GitHubでリポジトリを作成後、表示されるコマンドを実行：

```bash
# リモートリポジトリを追加
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# ブランチ名をmainに設定
git branch -M main

# リモートリポジトリにpush
git push -u origin main
```

### 3. 実際の例
ユーザー名が`johndoe`で、リポジトリ名が`titanic-ml-project`の場合：

```bash
git remote add origin https://github.com/johndoe/titanic-ml-project.git
git branch -M main
git push -u origin main
```

## 📁 含まれるファイル

このリポジトリには以下のファイルが含まれます：

- `README.md` - プロジェクトの概要
- `titanic_ml_project.md` - 詳細なプロジェクトドキュメント
- `create_submission.py` - 機械学習モデルスクリプト
- `titanic_submission.csv` - 提出用CSVファイル
- `titanic_submission.csv.zip` - 圧縮版
- `setup_remote.sh` - リモート設定手順
- `titanic/` - データセットファイル
- `.gitignore` - Git除外設定

## ⚠️ 注意事項

- `.DS_Store`、`__pycache__/`、`.venv/`などの設定ファイルは`.gitignore`で除外されています
- データセットファイル（`titanic/`フォルダ内）は含まれます
- 個人情報や機密情報は含まれていません

## 🎉 完了後

GitHubにpushが完了すると、以下のことができます：

1. コードの共有
2. バージョン管理
3. コラボレーション
4. プロジェクトの履歴追跡

## 🔗 便利なコマンド

```bash
# リモートリポジトリの確認
git remote -v

# 最新の変更をpush
git push

# リモートの変更をpull
git pull

# ブランチの確認
git branch -a
``` 