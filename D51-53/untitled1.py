# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:30:46 2019

@author: escc
"""


import pandas as pd
import numpy as np
import copy, time
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 檢查 DataFrame 空缺值的狀態
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))

data_path = '../csv/3rd-ml100marathon-midterm/'
df_train = pd.read_csv(data_path + 'train_data.csv')
df_test = pd.read_csv(data_path + 'test_features.csv')

train_Y = df_train['poi']
ids_train = df_train['name']
ids = df_test['name']
df_train = df_train.drop(['name', 'poi'] , axis=1)
df_test = df_test.drop(['name'] , axis=1)
df = pd.concat([df_train,df_test])
var_df_columns = df.columns
var_df_columns

var_df_dtypes = df.dtypes



#薪水
var_salary = ['salary']

#獎金
var_bonus = ['bonus']

#業務報銷
var_business_expenses_group = ['long_term_incentive','deferred_income','loan_advances'
        ,'deferral_payments','other','expenses']

#董事费
var_9 = ['director_fees']

#股權
var_10_11_12 = ['exercised_stock_options','restricted_stock','restricted_stock_deferred']
#股權 total
var_total_stock_value = ['total_stock_value']

df_cp = df.copy()

df_cp = df_cp.drop(['email_address'] , axis=1)

df_cp = df_cp.drop(var_10_11_12 , axis=1)

df_cp[var_total_stock_value] = df[var_total_stock_value].fillna(0)

df_cp[var_salary] = df[var_salary].fillna(df_cp[var_salary].median())
df_cp[var_bonus] = df[var_bonus].fillna(df_cp[var_bonus].median())

df_cp['business_expenses_total']=np.zeros([146])
for col in var_business_expenses_group:
    df_cp[col] = df_cp[col].fillna(0)
    df_cp['business_expenses_total'] = df_cp['business_expenses_total'] + df_cp[col]

df_cp = df_cp.drop(var_business_expenses_group , axis=1)   

for col_other in df_cp.columns:
    df_cp[col_other] = df_cp[col_other].fillna(0)


na_check(df_cp)  

# 將資料最大最小化
df_cp = MinMaxScaler().fit_transform(df_cp)  


# 將前述轉換完畢資料 df , 重新切成 train_X, test_X
train_num = train_Y.shape[0]
train_X = df_cp[:train_num]

update_test_X = df_cp[train_num:]
train_Y = train_Y.map(lambda x:1 if x==True else 0) 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=4)



# 使用三種模型 : 邏輯斯迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=11,
                                  max_depth=6, learning_rate=0.03)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
                            max_features='sqrt', max_depth=6, bootstrap=True)

# 線性迴歸預測檔 (結果有部分隨機, 請以 Kaggle 計算的得分為準, 以下模型同理)
lr.fit(train_X, train_Y)
lr_pred = lr.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': lr_pred})

#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test_lr.csv', index=False) 

# 梯度提升機預測檔 
gdbt.fit(x_train, y_train)
gdbt_pred = gdbt.predict_proba(x_test)[:,1]
from sklearn import datasets, metrics
print("bbbb",metrics.roc_auc_score(y_test, gdbt_pred))

# 隨機森林預測檔
rf.fit(train_X, train_Y)
rf_pred = rf.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': rf_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test_rf.csv', index=False)



from mlxtend.classifier import StackingClassifier
meta_estimator = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, 
                                           max_features='sqrt', max_depth=6, learning_rate=0.03)


stacking = StackingClassifier(classifiers=[gdbt, rf, lr],meta_classifier=meta_estimator)
stacking.fit(train_X, train_Y)
stacking_pred = stacking.predict(test_X)
sub = pd.DataFrame({'name': ids, 'poi': stacking_pred})
sub['poi'] = sub['poi'].map(lambda x:0.99 if x==True else 0.0) 
sub.to_csv('test_stacking.csv', index=False)

##
from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(20,300,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(tol=100, subsample=0.75, max_features=11,
                                  max_depth=6, learning_rate=0.03),param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)

gsearch1.fit(train_X,train_Y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(tol=100, subsample=0.75, max_features=11,
                                  max_depth=3,learning_rate=0.03,min_samples_split=2,n_estimators=250),param_grid = param_test2, scoring='roc_auc',iid=False,cv=5)

gsearch1


### gdbt2
gdbt2 = GradientBoostingClassifier(tol=100, subsample=0.75, max_features=11,
                                  max_depth=3,learning_rate=0.03,min_samples_split=2,n_estimators=250)

gdbt2.fit(train_X, train_Y)
gdbt_pred2 = gdbt2.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred2})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test_gdbt2.csv', index=False)
