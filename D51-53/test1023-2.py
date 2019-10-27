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

########## load data start
data_path = '../csv/3rd-ml100marathon-midterm/'
df_train = pd.read_csv(data_path + 'train_data.csv')
df_test = pd.read_csv(data_path + 'test_features.csv')

########## load data start end

########## 前處理理 (Processing) start
#薪水
var_salary = ['salary']

#獎金
var_bonus = ['bonus']

#業務報銷
var_business_expenses_group = ['long_term_incentive','deferred_income','loan_advances'
        ,'deferral_payments','other','expenses']

#董事费
var_director_fees = ['director_fees']

#總支出
var_total_payments = ['total_payments']

#股權
var_10_11_12 = ['exercised_stock_options','restricted_stock','restricted_stock_deferred']

#股權 total
var_total_stock_value = ['total_stock_value']

#email回覆
var_email_count_group = ['to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi','shared_receipt_with_poi']





#train_Y = df_train['poi']
train_Y = df_train['poi'].map(lambda x:1 if x==True else 0) 

ids_train = df_train['name']
ids = df_test['name']
df_train = df_train.drop(['name', 'poi'] , axis=1)
df_test = df_test.drop(['name'] , axis=1)
df = pd.concat([df_train,df_test])
var_df_columns = df.columns
var_df_columns

var_df_dtypes = df.dtypes




df_cp = df.copy()


df_cp = df_cp.drop(['email_address'] , axis=1)

## 股權計算 start
#df_cp = df_cp.drop(var_10_11_12 , axis=1)
df_cp[var_10_11_12] = df[var_10_11_12].fillna(0)

df_cp[var_total_stock_value] = df[var_total_stock_value].fillna(0)

## 股權計算 end

## 公司支出計算 strat
df_cp[var_salary] = df_cp[var_salary].fillna(df_cp[var_salary].median())
df_cp[var_bonus] = df_cp[var_bonus].fillna(df_cp[var_bonus].median())
df_cp[var_director_fees] = df_cp[var_director_fees].fillna(0)

df_cp.dtypes
df_cp['total_payments_new']=np.zeros([146])

df_cp['total_payments_new']= df_cp['total_payments_new'] + df_cp['salary']
df_cp['total_payments_new'] = df_cp['total_payments_new'] + df_cp['bonus']

for col in var_business_expenses_group:
    df_cp[col] = df_cp[col].fillna(0)
    df_cp['total_payments_new'] = df_cp['total_payments_new'] + df_cp[col]

df_cp['total_payments_new'] = df_cp['total_payments_new'] + df_cp['director_fees']

df_cp = df_cp.drop('total_payments' , axis=1)   


## 公司支出計算 end

## email 關係計算 start
for col_other in var_email_count_group:
    df_cp[col_other] = df_cp[col_other].fillna(df_cp[col_other].mean())

## email 關係計算 end
    
na_check(df_cp)


########## 前處理理 (Processing) end


# 將資料最大最小化
df_cp = MinMaxScaler().fit_transform(df_cp)  


# 將前述轉換完畢資料 df , 重新切成 train_X, test_X
train_num = train_Y.shape[0]
train_X = df_cp[:train_num]
test_X = df_cp[train_num:]

# 使用三種模型 : 邏輯斯迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=19,
                                  max_depth=3, learning_rate=0.03)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
                            max_features='sqrt', max_depth=6, bootstrap=True)

# 線性迴歸預測檔 (結果有部分隨機, 請以 Kaggle 計算的得分為準, 以下模型同理)
lr.fit(train_X, train_Y)
lr_pred = lr.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': lr_pred})

#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test1023_lr.csv', index=False) 

# 梯度提升機預測檔 
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test_gdbt1023--.csv', index=False)

# 隨機森林預測檔
rf.fit(train_X, train_Y)
rf_pred = rf.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': rf_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('test1023_rf.csv', index=False)

# 混合泛化預測檔 
"""
Your Code Here
"""
blending_pred = lr_pred*0.3  + gdbt_pred*0.4 + rf_pred*0.3
sub = pd.DataFrame({'name': ids, 'poi': blending_pred})

sub.to_csv('test1023_blending.csv', index=False)



