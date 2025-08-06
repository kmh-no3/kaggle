#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jupyter Notebookファイルを生成するスクリプト
"""

import json
import nbformat as nbf

def create_simple_notebook():
    """シンプルなCSVビューアーのノートブックを作成"""
    
    # 新しいノートブックを作成
    nb = nbf.v4.new_notebook()
    
    # セルを追加
    cells = []
    
    # タイトルセル
    title_cell = nbf.v4.new_markdown_cell("""# 📊 CSVファイルビューアー

TitanicデータセットのCSVファイルを簡単に表示します。""")
    cells.append(title_cell)
    
    # ライブラリインポート
    import_cell = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

print("✅ ライブラリ読み込み完了")""")
    cells.append(import_cell)
    
    # 訓練データセクション
    train_section = nbf.v4.new_markdown_cell("""## 1. 訓練データ (train.csv)""")
    cells.append(train_section)
    
    train_load = nbf.v4.new_code_cell("""# 訓練データの読み込み
train_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/train.csv')

print(f"📊 データサイズ: {train_df.shape[0]}行 × {train_df.shape[1]}列")
print(f"📋 列名: {list(train_df.columns)}")
print("\\n" + "="*80)

# 最初の10行を表示
train_df.head(10)""")
    cells.append(train_load)
    
    train_stats = nbf.v4.new_code_cell("""# 基本統計情報
print("📈 基本統計情報")
print("="*30)
train_df.describe()""")
    cells.append(train_stats)
    
    train_missing = nbf.v4.new_code_cell("""# 欠損値の確認
missing = train_df.isnull().sum()
print("⚠️ 欠損値の確認")
print("="*30)
for col, count in missing.items():
    if count > 0:
        percentage = (count / len(train_df)) * 100
        print(f"{col}: {count}件 ({percentage:.1f}%)")
    else:
        print(f"{col}: 欠損なし")""")
    cells.append(train_missing)
    
    # テストデータセクション
    test_section = nbf.v4.new_markdown_cell("""## 2. テストデータ (test.csv)""")
    cells.append(test_section)
    
    test_load = nbf.v4.new_code_cell("""# テストデータの読み込み
test_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/test.csv')

print(f"📊 データサイズ: {test_df.shape[0]}行 × {test_df.shape[1]}列")
print(f"📋 列名: {list(test_df.columns)}")
print("\\n" + "="*80)

# 最初の10行を表示
test_df.head(10)""")
    cells.append(test_load)
    
    test_missing = nbf.v4.new_code_cell("""# テストデータの欠損値
missing_test = test_df.isnull().sum()
print("⚠️ テストデータの欠損値")
print("="*30)
for col, count in missing_test.items():
    if count > 0:
        percentage = (count / len(test_df)) * 100
        print(f"{col}: {count}件 ({percentage:.1f}%)")
    else:
        print(f"{col}: 欠損なし")""")
    cells.append(test_missing)
    
    # サンプル提出セクション
    submission_section = nbf.v4.new_markdown_cell("""## 3. サンプル提出ファイル (gender_submission.csv)""")
    cells.append(submission_section)
    
    submission_load = nbf.v4.new_code_cell("""# サンプル提出ファイルの読み込み
gender_submission = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/gender_submission.csv')

print(f"📊 データサイズ: {gender_submission.shape[0]}行 × {gender_submission.shape[1]}列")
print(f"📋 列名: {list(gender_submission.columns)}")
print("\\n" + "="*80)

# 最初の10行を表示
gender_submission.head(10)""")
    cells.append(submission_load)
    
    submission_stats = nbf.v4.new_code_cell("""# サンプル提出の統計
print("📈 サンプル提出の統計")
print("="*30)
print(f"生存予測: {gender_submission['Survived'].sum()}人")
print(f"死亡予測: {(gender_submission['Survived'] == 0).sum()}人")
print(f"生存率予測: {gender_submission['Survived'].mean():.3f} ({gender_submission['Survived'].mean()*100:.1f}%)")""")
    cells.append(submission_stats)
    
    # 比較セクション
    compare_section = nbf.v4.new_markdown_cell("""## 4. データの比較""")
    cells.append(compare_section)
    
    compare_cell = nbf.v4.new_code_cell("""print("📊 データセットの比較")
print("="*50)
print(f"訓練データ: {train_df.shape[0]}行 × {train_df.shape[1]}列")
print(f"テストデータ: {test_df.shape[0]}行 × {test_df.shape[1]}列")
print(f"サンプル提出: {gender_submission.shape[0]}行 × {gender_submission.shape[1]}列")

print("\\n🔍 主な違い:")
print("• 訓練データには 'Survived' 列がある（教師データ）")
print("• テストデータには 'Survived' 列がない（予測対象）")
print("• サンプル提出は性別ベースの簡単な予測例")""")
    cells.append(compare_cell)
    
    # セルをノートブックに追加
    nb.cells = cells
    
    # ノートブックを保存
    with open('simple_csv_viewer.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("✅ simple_csv_viewer.ipynb を作成しました")

def create_analysis_notebook():
    """詳細分析用のノートブックを作成"""
    
    # 新しいノートブックを作成
    nb = nbf.v4.new_notebook()
    
    # セルを追加
    cells = []
    
    # タイトルセル
    title_cell = nbf.v4.new_markdown_cell("""# 🚢 Titanicデータセット分析

このノートブックでは、Titanicの乗客データを分析して、生存予測のためのデータ探索を行います。""")
    cells.append(title_cell)
    
    # ライブラリインポート
    import_cell = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')

print("✅ ライブラリのインポート完了")""")
    cells.append(import_cell)
    
    # データ読み込み
    data_section = nbf.v4.new_markdown_cell("""## 2. データの読み込み""")
    cells.append(data_section)
    
    data_load = nbf.v4.new_code_cell("""# データの読み込み
train_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/train.csv')
test_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/test.csv')
gender_submission = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/gender_submission.csv')

print("📊 データ読み込み完了")
print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")
print(f"サンプル提出: {gender_submission.shape}")""")
    cells.append(data_load)
    
    # 基本情報
    info_section = nbf.v4.new_markdown_cell("""## 3. 訓練データの基本情報""")
    cells.append(info_section)
    
    info_cell = nbf.v4.new_code_cell("""print("🔍 訓練データの基本情報")
print("="*50)
train_df.info()""")
    cells.append(info_cell)
    
    head_cell = nbf.v4.new_code_cell("""print("📋 訓練データの最初の5行")
print("="*50)
train_df.head()""")
    cells.append(head_cell)
    
    describe_cell = nbf.v4.new_code_cell("""print("📊 数値データの統計情報")
print("="*50)
train_df.describe()""")
    cells.append(describe_cell)
    
    # 欠損値
    missing_section = nbf.v4.new_markdown_cell("""## 4. 欠損値の確認""")
    cells.append(missing_section)
    
    missing_cell = nbf.v4.new_code_cell("""# 欠損値の確認
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("⚠️ 訓練データの欠損値")
print("="*30)
for col, count in missing_train.items():
    if count > 0:
        percentage = (count / len(train_df)) * 100
        print(f"{col}: {count}件 ({percentage:.1f}%)")

print("\\n⚠️ テストデータの欠損値")
print("="*30)
for col, count in missing_test.items():
    if count > 0:
        percentage = (count / len(test_df)) * 100
        print(f"{col}: {count}件 ({percentage:.1f}%)")""")
    cells.append(missing_cell)
    
    # 生存率分析
    survival_section = nbf.v4.new_markdown_cell("""## 5. 生存率の分析""")
    cells.append(survival_section)
    
    survival_cell = nbf.v4.new_code_cell("""# 全体の生存率
survival_rate = train_df['Survived'].mean()
print(f"📈 全体の生存率: {survival_rate:.3f} ({survival_rate*100:.1f}%)")

# 性別別の生存率
gender_survival = train_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
gender_survival.columns = ['総数', '生存者数', '生存率']
print("\\n👥 性別別の生存率")
print("="*30)
print(gender_survival)""")
    cells.append(survival_cell)
    
    # セルをノートブックに追加
    nb.cells = cells
    
    # ノートブックを保存
    with open('titanic_analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("✅ titanic_analysis.ipynb を作成しました")

if __name__ == "__main__":
    print("📝 Jupyter Notebookファイルを作成中...")
    create_simple_notebook()
    create_analysis_notebook()
    print("🎉 完了しました！") 