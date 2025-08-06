#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
予測結果の詳細分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_predictions():
    """予測結果を詳細に分析"""
    
    # 予測結果を読み込み
    df = pd.read_csv('titanic_predictions_gradient_boosting.csv')
    
    print("📊 予測結果の詳細分析")
    print("="*50)
    
    # 基本統計
    print(f"総乗客数: {len(df)}人")
    print(f"生存予測: {df['Survived'].sum()}人")
    print(f"死亡予測: {(df['Survived'] == 0).sum()}人")
    print(f"生存率: {df['Survived'].mean():.3f} ({df['Survived'].mean()*100:.1f}%)")
    
    # 性別別分析
    print("\n👥 性別別分析:")
    print("="*30)
    gender_analysis = df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean'])
    gender_analysis.columns = ['総数', '生存者数', '生存率']
    print(gender_analysis)
    
    # 客室クラス別分析
    print("\n🎫 客室クラス別分析:")
    print("="*30)
    class_analysis = df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean'])
    class_analysis.columns = ['総数', '生存者数', '生存率']
    print(class_analysis)
    
    # 乗船港別分析
    print("\n🚢 乗船港別分析:")
    print("="*30)
    embarked_analysis = df.groupby('Embarked')['Survived'].agg(['count', 'sum', 'mean'])
    embarked_analysis.columns = ['総数', '生存者数', '生存率']
    embarked_analysis.index = embarked_analysis.index.map({
        'S': 'サウサンプトン', 
        'C': 'シェルブール', 
        'Q': 'クイーンズタウン'
    })
    print(embarked_analysis)
    
    # 年齢グループ別分析
    print("\n👶 年齢グループ別分析:")
    print("="*30)
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 30, 50, 100], 
                           labels=['子供(0-12)', '青年(13-18)', '若年(19-30)', '中年(31-50)', '高齢(51+)'])
    age_analysis = df.groupby('AgeGroup')['Survived'].agg(['count', 'sum', 'mean'])
    age_analysis.columns = ['総数', '生存者数', '生存率']
    print(age_analysis)
    
    # 家族サイズ別分析
    print("\n👨‍👩‍👧‍👦 家族サイズ別分析:")
    print("="*30)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_analysis = df.groupby('FamilySize')['Survived'].agg(['count', 'sum', 'mean'])
    family_analysis.columns = ['総数', '生存者数', '生存率']
    print(family_analysis.head(10))
    
    # 運賃グループ別分析
    print("\n💰 運賃グループ別分析:")
    print("="*30)
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['低運賃', '中低運賃', '中高運賃', '高運賃'], duplicates='drop')
    fare_analysis = df.groupby('FareGroup')['Survived'].agg(['count', 'sum', 'mean'])
    fare_analysis.columns = ['総数', '生存者数', '生存率']
    print(fare_analysis)
    
    # 客室の有無別分析
    print("\n🏠 客室の有無別分析:")
    print("="*30)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    cabin_analysis = df.groupby('HasCabin')['Survived'].agg(['count', 'sum', 'mean'])
    cabin_analysis.columns = ['総数', '生存者数', '生存率']
    cabin_analysis.index = cabin_analysis.index.map({0: '客室なし', 1: '客室あり'})
    print(cabin_analysis)
    
    # 最も生存率が高いグループ
    print("\n🏆 最も生存率が高いグループ:")
    print("="*30)
    
    # 性別×客室クラスの組み合わせ
    combo_analysis = df.groupby(['Sex', 'Pclass'])['Survived'].mean().sort_values(ascending=False)
    print("性別×客室クラス別生存率（上位5位）:")
    for (sex, pclass), rate in combo_analysis.head().items():
        sex_jp = '女性' if sex == 'female' else '男性'
        class_jp = f'{pclass}等客室'
        print(f"  {sex_jp} + {class_jp}: {rate:.3f} ({rate*100:.1f}%)")
    
    # 最も生存率が低いグループ
    print("\n💀 最も生存率が低いグループ:")
    print("="*30)
    print("性別×客室クラス別生存率（下位5位）:")
    for (sex, pclass), rate in combo_analysis.tail().items():
        sex_jp = '女性' if sex == 'female' else '男性'
        class_jp = f'{pclass}等客室'
        print(f"  {sex_jp} + {class_jp}: {rate:.3f} ({rate*100:.1f}%)")
    
    # 生存予測の詳細例
    print("\n📋 生存予測の詳細例（上位10件）:")
    print("="*30)
    survivors = df[df['Survived'] == 1].head(10)
    for _, row in survivors.iterrows():
        sex_jp = '女性' if row['Sex'] == 'female' else '男性'
        embarked_jp = {'S': 'サウサンプトン', 'C': 'シェルブール', 'Q': 'クイーンズタウン'}[row['Embarked']]
        print(f"  {row['Name']} ({sex_jp}, {row['Age']}歳, {row['Pclass']}等客室, {embarked_jp})")
    
    print("\n📋 死亡予測の詳細例（上位10件）:")
    print("="*30)
    non_survivors = df[df['Survived'] == 0].head(10)
    for _, row in non_survivors.iterrows():
        sex_jp = '女性' if row['Sex'] == 'female' else '男性'
        embarked_jp = {'S': 'サウサンプトン', 'C': 'シェルブール', 'Q': 'クイーンズタウン'}[row['Embarked']]
        print(f"  {row['Name']} ({sex_jp}, {row['Age']}歳, {row['Pclass']}等客室, {embarked_jp})")
    
    print("\n🎉 分析完了！")

if __name__ == "__main__":
    analyze_predictions() 