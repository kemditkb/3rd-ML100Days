# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:08:52 2019

@author: escc1122
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_path = './'
df_train_org = pd.read_csv(data_path + 'train.csv',header=None )
df_test_org = pd.read_csv(data_path + 'test.csv',header=None )
df_train_y = pd.read_csv(data_path + 'trainLabels.csv',header=None )


df_train_org.head()
x_train, x_test, y_train, y_test = train_test_split(df_train_org, df_train_y, test_size=0.25, random_state=4)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gdbt = GradientBoostingClassifier(learning_rate=0.01)


# 訓練模型
gdbt.fit(x_train, y_train)

y_pred = gdbt.predict(x_test)


from sklearn.metrics import confusion_matrix

var_confusion_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)

acc = accuracy_score(y_test, y_pred)


y_pred_send = gdbt.predict(df_test_org)

ids = range(1, 9001, 1)


var_a = pd.DataFrame({'Id':ids,'Solution': y_pred_send})


var_a.to_csv('test_gbdt.csv', index=False)