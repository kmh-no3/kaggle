#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプルなCSVビューアー
"""

import pandas as pd
import os

def display_csv_simple(file_path, rows=10):
    """CSVファイルをシンプルに表示"""
    try:
        df = pd.read_csv(file_path)
        
        print(f"\n📊 {os.path.basename(file_path)}")
        print(f"📈 データサイズ: {len(df)}行 × {len(df.columns)}列")
        print(f"📋 列名: {', '.join(df.columns.tolist())}")
        print("\n" + "="*80)
        
        # 最初の数行を表示
        print(df.head(rows).to_string(index=False))
        
        # 欠損値の情報
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  欠損値:")
            for col, count in missing.items():
                if count > 0:
                    print(f"   {col}: {count}件")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ エラー: {e}")

def main():
    """メイン関数"""
    files = [
        "Titanic - Machine Learning from Disaster/titanic/train.csv",
        "Titanic - Machine Learning from Disaster/titanic/test.csv", 
        "Titanic - Machine Learning from Disaster/titanic/gender_submission.csv"
    ]
    
    print("🚢 Titanicデータセット CSVビューアー")
    print("="*50)
    
    for file_path in files:
        if os.path.exists(file_path):
            display_csv_simple(file_path)
            input("\n⏸️  Enterキーを押して次へ...")

if __name__ == "__main__":
    main() 