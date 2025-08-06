#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titanic生存予測モデル - 提出用CSV作成
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_training=True):
    """データの前処理を行う関数"""
    
    # データのコピーを作成
    df_processed = df.copy()
    
    # 1. 年齢の欠損値を中央値で補完
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    
    # 2. 運賃の欠損値を中央値で補完（テストデータのみ）
    if 'Fare' in df_processed.columns:
        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
    
    # 3. 乗船港の欠損値を最頻値で補完
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    
    # 4. 客室番号の欠損値を処理（欠損の場合は0、そうでなければ1）
    df_processed['HasCabin'] = df_processed['Cabin'].notna().astype(int)
    
    # 5. 家族サイズの計算
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    # 6. 一人旅かどうかの判定
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    # 7. 年齢グループの作成
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                      bins=[0, 12, 18, 30, 50, 100], 
                                      labels=[0, 1, 2, 3, 4])
    
    # 8. 運賃グループの作成
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], 
                                       q=4, 
                                       labels=[0, 1, 2, 3], 
                                       duplicates='drop')
    
    # 9. 性別のエンコーディング
    df_processed['Sex_encoded'] = (df_processed['Sex'] == 'female').astype(int)
    
    # 10. 乗船港のエンコーディング
    embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
    df_processed['Embarked_encoded'] = df_processed['Embarked'].map(embarked_mapping)
    
    # 使用する特徴量を選択
    features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked_encoded', 'HasCabin', 'FamilySize', 'IsAlone', 
                'AgeGroup', 'FareGroup']
    
    # 欠損値を含む列を除外
    available_features = [f for f in features if f in df_processed.columns]
    
    if is_training:
        return df_processed[available_features], df_processed['Survived']
    else:
        return df_processed[available_features]

def main():
    """メイン関数"""
    
    print("🚢 Titanic生存予測モデル - 提出用CSV作成")
    print("="*50)
    
    # 1. データの読み込み
    print("📊 データを読み込み中...")
    train_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/train.csv')
    test_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/test.csv')
    
    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")
    
    # 2. データの前処理
    print("\n🔧 データの前処理中...")
    X_train, y_train = preprocess_data(train_df, is_training=True)
    X_test = preprocess_data(test_df, is_training=False)
    
    print(f"前処理後の特徴量: {X_train.shape}")
    
    # 3. モデルの訓練
    print("\n🤖 モデルを訓練中...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. テストデータで予測
    print("\n🔮 テストデータで予測中...")
    predictions = model.predict(X_test)
    
    print(f"予測完了: {len(predictions)}件")
    print(f"生存予測: {predictions.sum()}人")
    print(f"死亡予測: {(predictions == 0).sum()}人")
    print(f"生存率予測: {predictions.mean():.3f} ({predictions.mean()*100:.1f}%)")
    
    # 5. 提出用CSVファイルの作成
    print("\n💾 提出用CSVファイルを作成中...")
    
    # PassengerIdとSurvivedのみのデータフレームを作成
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # ファイルを保存
    submission_df.to_csv('titanic_submission.csv', index=False)
    
    print("✅ titanic_submission.csv を作成しました")
    print(f"データサイズ: {submission_df.shape}")
    
    # 6. 結果の確認
    print("\n📋 提出用ファイルの最初の10行:")
    print(submission_df.head(10))
    
    print("\n📊 予測結果の概要:")
    print(f"総乗客数: {len(submission_df)}人")
    print(f"生存予測: {submission_df['Survived'].sum()}人")
    print(f"死亡予測: {(submission_df['Survived'] == 0).sum()}人")
    print(f"生存率: {submission_df['Survived'].mean():.3f} ({submission_df['Survived'].mean()*100:.1f}%)")
    
    print("\n🎉 完了しました！")

if __name__ == "__main__":
    main() 