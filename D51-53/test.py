# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 01:14:49 2019

@author: escc
"""

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = '../csv/3rd-ml100marathon-midterm/'
df = pd.read_csv(data_path + 'train_data.csv')
df.dtypes
df.columns


var_cloume = ['name','poi']

df[var_cloume].head(100)

# 計算df整體相關係數, 並繪製成熱圖
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
sns.heatmap(corr)
plt.show()

high_list = list(corr[(corr['poi']>0.4) | (corr['poi']<-0.4)].index)
high_list.pop(2)

df_high = df[high_list]

df[high_list].head(100)
# 檢查 DataFrame 空缺值的狀態
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))
    
na_check(df_high)

# 部分欄位缺值填補 0
zero_cols = ['loan_advances']
for col in zero_cols:
    df_high[col] = df_high[col].fillna(0)
    
# 部分欄位缺值補眾數
mode_cols = ['exercised_stock_options', 'total_stock_value']
for col in mode_cols:
    df_high[col] = df_high[col].fillna(df_high[col].mode()[0])    

na_check(df_high)

# 將資料最大最小化
from sklearn.preprocessing import MinMaxScaler
df_high = MinMaxScaler().fit_transform(df_high)

#df_high.as_matrix()

df_high.columns



from sklearn.model_selection import train_test_split, KFold, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(df_high, df['poi'], test_size=0.25, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn import metrics
linear = LinearRegression(normalize=False, fit_intercept=True, copy_X=True)
linear.fit(x_train, y_train)
linear_pred = linear.predict(x_test)
metrics.mean_squared_error(y_test, linear_pred)


# 建立模型
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)
metrics.mean_squared_error(y_test, lr_pred)


