#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度の検証と答え合わせ
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    
    print("🔍 精度の検証と答え合わせ")
    print("="*50)
    
    # 1. データの読み込み
    print("📊 データを読み込み中...")
    train_df = pd.read_csv('Titanic - Machine Learning from Disaster/titanic/train.csv')
    
    print(f"訓練データ: {train_df.shape}")
    
    # 2. データの前処理
    print("\n🔧 データの前処理中...")
    X_train, y_train = preprocess_data(train_df, is_training=True)
    
    print(f"前処理後の特徴量: {X_train.shape}")
    
    # 3. データの分割（検証用）
    print("\n📊 データを分割中...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"訓練セット: {X_train_split.shape}")
    print(f"検証セット: {X_val.shape}")
    
    # 4. モデルの訓練
    print("\n🤖 モデルを訓練中...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)
    
    # 5. 検証セットで予測
    print("\n🔮 検証セットで予測中...")
    y_pred = model.predict(X_val)
    
    # 6. 精度の計算
    accuracy = accuracy_score(y_val, y_pred)
    print(f"精度: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 7. 詳細な分類レポート
    print("\n📋 詳細な分類レポート:")
    print("="*30)
    print(classification_report(y_val, y_pred, target_names=['死亡', '生存']))
    
    # 8. 混同行列
    print("\n📊 混同行列:")
    print("="*30)
    cm = confusion_matrix(y_val, y_pred)
    print("予測\\実際    死亡  生存")
    print("死亡        {:4d}  {:4d}".format(cm[0,0], cm[0,1]))
    print("生存        {:4d}  {:4d}".format(cm[1,0], cm[1,1]))
    
    # 9. 具体的な答え合わせ例
    print("\n🔍 具体的な答え合わせ例（検証セットから10件）:")
    print("="*50)
    
    # 検証セットの元データを取得
    val_indices = train_df.index[train_test_split(
        range(len(train_df)), train_df['Survived'], 
        test_size=0.2, random_state=42, stratify=train_df['Survived']
    )[1]]
    
    val_data = train_df.iloc[val_indices].copy()
    val_data['Predicted'] = y_pred
    val_data['Actual'] = y_val.values
    val_data['Correct'] = (val_data['Predicted'] == val_data['Actual'])
    
    # 正解と不正解の例を表示
    print("\n✅ 正解した予測の例:")
    correct_examples = val_data[val_data['Correct'] == True].head(5)
    for _, row in correct_examples.iterrows():
        sex_jp = '女性' if row['Sex'] == 'female' else '男性'
        embarked_jp = {'S': 'サウサンプトン', 'C': 'シェルブール', 'Q': 'クイーンズタウン'}[row['Embarked']]
        result = '生存' if row['Actual'] == 1 else '死亡'
        print(f"  {row['Name']} ({sex_jp}, {row['Age']}歳, {row['Pclass']}等客室) → {result} (正解)")
    
    print("\n❌ 間違った予測の例:")
    incorrect_examples = val_data[val_data['Correct'] == False].head(5)
    for _, row in incorrect_examples.iterrows():
        sex_jp = '女性' if row['Sex'] == 'female' else '男性'
        embarked_jp = {'S': 'サウサンプトン', 'C': 'シェルブール', 'Q': 'クイーンズタウン'}[row['Embarked']]
        predicted = '生存' if row['Predicted'] == 1 else '死亡'
        actual = '生存' if row['Actual'] == 1 else '死亡'
        print(f"  {row['Name']} ({sex_jp}, {row['Age']}歳, {row['Pclass']}等客室) → 予測:{predicted}, 実際:{actual}")
    
    # 10. 性別別の精度
    print("\n👥 性別別の精度:")
    print("="*30)
    val_data_with_sex = val_data.copy()
    val_data_with_sex['Sex_encoded'] = (val_data_with_sex['Sex'] == 'female').astype(int)
    
    for sex, sex_name in [(0, '男性'), (1, '女性')]:
        sex_data = val_data_with_sex[val_data_with_sex['Sex_encoded'] == sex]
        if len(sex_data) > 0:
            sex_accuracy = sex_data['Correct'].mean()
            print(f"{sex_name}: {sex_accuracy:.4f} ({sex_accuracy*100:.2f}%) - {len(sex_data)}件")
    
    # 11. 客室クラス別の精度
    print("\n🎫 客室クラス別の精度:")
    print("="*30)
    for pclass in [1, 2, 3]:
        class_data = val_data[val_data['Pclass'] == pclass]
        if len(class_data) > 0:
            class_accuracy = class_data['Correct'].mean()
            print(f"{pclass}等客室: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {len(class_data)}件")
    
    print("\n🎉 検証完了！")

if __name__ == "__main__":
    main() 