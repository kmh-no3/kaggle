#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSVファイルを表形式で表示するスクリプト
"""

import pandas as pd
import sys
import os

def view_csv(file_path, max_rows=20, max_cols=None):
    """
    CSVファイルを表形式で表示する関数
    
    Args:
        file_path (str): CSVファイルのパス
        max_rows (int): 表示する最大行数（デフォルト: 20）
        max_cols (int): 表示する最大列数（デフォルト: None = 全列）
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        print(f"\n{'='*60}")
        print(f"ファイル: {file_path}")
        print(f"行数: {len(df)}, 列数: {len(df.columns)}")
        print(f"{'='*60}")
        
        # 列名を表示
        print("\n列名:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        # データの最初の数行を表示
        print(f"\n最初の{min(max_rows, len(df))}行:")
        print(df.head(max_rows).to_string(index=True, max_cols=max_cols))
        
        # データの基本情報を表示
        print(f"\nデータの基本情報:")
        print(df.info())
        
        # 数値列の統計情報を表示
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n数値列の統計情報:")
            print(df[numeric_cols].describe())
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"エラー: {e}")

def main():
    """メイン関数"""
    # 利用可能なCSVファイルのリスト
    csv_files = [
        "Titanic - Machine Learning from Disaster/titanic/train.csv",
        "Titanic - Machine Learning from Disaster/titanic/test.csv",
        "Titanic - Machine Learning from Disaster/titanic/gender_submission.csv"
    ]
    
    print("利用可能なCSVファイル:")
    for i, file_path in enumerate(csv_files, 1):
        if os.path.exists(file_path):
            print(f"  {i}. {file_path}")
    
    print("\nどのファイルを表示しますか？")
    print("1: train.csv (訓練データ)")
    print("2: test.csv (テストデータ)")
    print("3: gender_submission.csv (性別予測サンプル)")
    print("4: すべて表示")
    
    try:
        choice = input("\n選択してください (1-4): ").strip()
        
        if choice == "1":
            view_csv(csv_files[0])
        elif choice == "2":
            view_csv(csv_files[1])
        elif choice == "3":
            view_csv(csv_files[2])
        elif choice == "4":
            for file_path in csv_files:
                if os.path.exists(file_path):
                    view_csv(file_path)
                    input("\nEnterキーを押して次のファイルを表示...")
        else:
            print("無効な選択です。")
            
    except KeyboardInterrupt:
        print("\n\nプログラムを終了します。")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main() 