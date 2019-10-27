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
#df_cp = df_cp.drop(var_business_expenses_group , axis=1)   

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

update_test_X = df_cp[train_num:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=4)


########## model start
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

gdbt = GradientBoostingRegressor(tol=100, subsample=0.75, n_estimators=250, max_features=19, 
                                 max_depth=6, learning_rate=0.03)
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict(test_X)
sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})
sub.to_csv('test1023_gdbt.csv', index=False)

gdbt_pred_check = gdbt.predict(train_X)




#####
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import datasets, metrics
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=19, 
                                 max_depth=6, learning_rate=0.03)

gdbt.fit(x_train, y_train)
gdbt_pred = gdbt.predict_proba(x_test)[:,1]
#sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})
#sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
#sub.to_csv('test1023_gdbt_c.csv', index=False)

y_pred= gdbt.predict(x_test)
y_predprob= gdbt.predict_proba(x_test)[:,1]
print("aaaa",metrics.accuracy_score(y_test.values, y_pred))

print("bbbb",metrics.roc_auc_score(y_test, y_predprob))

#print("Acuuracy: ", acc)
from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(20,300,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(tol=100, subsample=0.75, max_features=19,
                                  max_depth=6, learning_rate=0.03),param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)

gsearch1.fit(x_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(tol=100, subsample=0.75, max_features=11,
                                  learning_rate=0.03,n_estimators=120),param_grid = param_test2, scoring='roc_auc',iid=False,cv=5)

gsearch1.fit(x_train,y_train)




gdbt = GradientBoostingClassifier()

gdbt.fit(x_train, y_train)
gdbt_pred = gdbt.predict_proba(update_test_X)[:,1]
sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})

sub.to_csv('test1023_gdbt2_c.csv', index=False)

y_pred= gdbt.predict(x_test)
y_predprob= gdbt.predict_proba(x_test)[:,1]
print("aaaa",metrics.accuracy_score(y_test.values, y_pred))

print("bbbb",metrics.roc_auc_score(y_test, y_predprob))
########## model end










