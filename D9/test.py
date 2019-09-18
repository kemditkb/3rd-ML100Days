# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:52:02 2019

@author: escc
"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%matplotlib inline

# 設定 data_path
dir_data = '../csv'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

# 先篩選數值型的欄位
"""
YOUR CODE HERE, fill correct data types (for example str, float, int, ...)
"""
dtype_select = [np.dtype('int64'), np.dtype('float64')]

var_is_in_array = app_train.dtypes.isin(dtype_select)

numeric_columns = list(app_train.columns[list(var_is_in_array)])

# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns %s" % len(numeric_columns))
app_train['SK_ID_CURR'].describe()
# 檢視這些欄位的數值範圍
for col in numeric_columns:
    """
    Your CODE HERE, make the box plot
    """
    print((app_train[col]).describe())
    plt.hist(app_train[col])
    plt.show()
    
 

cdf = app_train['AMT_INCOME_TOTAL']
cdf2 = cdf.sort_values()


plt.plot(list(cdf2), cdf.index/cdf.index.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
plt.plot(np.log(list(cdf2)), cdf.index/cdf.index.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()
    
    
    
def ecdf(data):
   n = len(data)
   x = np.log(np.sort(data))
   y = np.arange(1, n+1) / n
   return x, y

x, y = ecdf(app_train['AMT_INCOME_TOTAL'])
plt.plot(x, y)
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.show()   
    
    
    
    