#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titanic生存予測モデル
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
    
    print("🚢 Titanic生存予測モデルを開始します")
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
    print(f"使用する特徴量: {list(X_train.columns)}")
    
    # 3. データの分割
    print("\n📊 データを分割中...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"訓練セット: {X_train_split.shape}")
    print(f"検証セット: {X_val.shape}")
    
    # 4. 複数のモデルを試す
    print("\n🤖 複数のモデルを訓練中...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 {name} を訓練中...")
        
        # モデルを訓練
        model.fit(X_train_split, y_train_split)
        
        # 予測
        y_pred = model.predict(X_val)
        
        # 精度を計算
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy
        }
        
        print(f"精度: {accuracy:.4f}")
        
        # クロスバリデーション
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"クロスバリデーション精度: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 5. 最良のモデルを選択
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 最良のモデル: {best_model_name}")
    print(f"精度: {best_accuracy:.4f}")
    
    # 6. テストデータで予測
    print("\n🔮 テストデータで予測中...")
    predictions = best_model.predict(X_test)
    
    print(f"予測完了: {len(predictions)}件")
    print(f"生存予測: {predictions.sum()}人")
    print(f"死亡予測: {(predictions == 0).sum()}人")
    print(f"生存率予測: {predictions.mean():.3f} ({predictions.mean()*100:.1f}%)")
    
    # 7. 結果の保存
    print("\n💾 結果を保存中...")
    test_df_with_predictions = test_df.copy()
    test_df_with_predictions['Survived'] = predictions
    
    output_filename = f'titanic_predictions_{best_model_name.replace(" ", "_").lower()}.csv'
    test_df_with_predictions.to_csv(output_filename, index=False)
    
    print(f"予測結果を保存しました: {output_filename}")
    
    # 8. 特徴量の重要度（Random Forestの場合）
    if isinstance(best_model, RandomForestClassifier):
        print("\n📊 特徴量の重要度:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 9. データの意味を表示
    print("\n📋 データの意味:")
    print("• Survived: 0=死亡, 1=生存")
    print("• Pclass: 1=1等客室, 2=2等客室, 3=3等客室")
    print("• Embarked: C=シェルブール, Q=クイーンズタウン, S=サウサンプトン")
    print("• Sex: male=男性, female=女性")
    print("• Age: 年齢")
    print("• SibSp: 兄弟姉妹・配偶者の数")
    print("• Parch: 親・子供の数")
    print("• Fare: 運賃")
    print("• Cabin: 客室番号")
    print("• Ticket: チケット番号")
    
    print("\n🎉 完了しました！")

if __name__ == "__main__":
    main() 