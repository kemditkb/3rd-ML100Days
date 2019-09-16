# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:48:50 2019

@author: escc
"""

import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = '../csv/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

app_train.dtypes.value_counts()

app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)

"""
Label encoding

有仔細閱讀參考資料的人可以發現，Label encoding 的表示方式會讓同一個欄位底下的類別之間有大小關係 
(0<1<2<...)，所以在這裡我們只對有類別數量小於等於 2 的類別型欄位示範使用 Label encoding，但不表示這樣處理是最好的，一切取決於欄位本身的意義適合哪一種表示方法
"""
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()
le_count = 0

var_head_before = app_train.head(5);



# Iterate through the columns
"""
https://blog.csdn.net/quintind/article/details/79850455
1、LabelEncoder

LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：

fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
inverse_transform(y)：根据索引值y获得原始数据。
transform(y) ：将y转变成索引值。
"""
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            print (app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
var_head_after = app_train.head(5);


"""
One Hot encoding
"""
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())

#==============================

# 設定 data_path, 並讀取 app_train
dir_data = '../csv/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()

var_sub_train_one_hot_encoding = pd.get_dummies(sub_train);
print(var_sub_train_one_hot_encoding.shape);
var_sub_train_one_hot_encoding.head()





