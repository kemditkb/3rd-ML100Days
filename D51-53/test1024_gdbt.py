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
df_train_tmp = pd.read_csv(data_path + 'train_data.csv')
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
train_Y = df_train_tmp['poi'].map(lambda x:1 if x==True else 0) 

ids_train = df_train_tmp['name']
ids = df_test['name']
df_train = df_train_tmp.drop(['name', 'poi'] , axis=1)
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

feat_labels = df_cp.columns
feat_labels
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
from sklearn.ensemble import GradientBoostingClassifier
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=13, 
                                 max_depth=4, learning_rate=0.01)


# 訓練模型
gdbt.fit(x_train, y_train)

# 預測測試集
y_pred = gdbt.predict(x_test)


########## model end



########## 糢型憑估  start
from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
check_view = pd.DataFrame({'pred_poi': y_pred,'poi':y_test})
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
var_confusion_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)

importances = gdbt.feature_importances_
indices = np.argsort(importances)

for f in range(x_train.shape[1]):
    print(f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
########## 糢型憑估  end


    #print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
threshold =0.04
x_selected = x_train[:, importances > threshold]
x_selected_test = x_test[:, importances > threshold]
x_selected.shape 
x_selected_update = update_test_X[:, importances > threshold]

gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=x_selected.shape[1], 
                                 max_depth=4, learning_rate=0.01)

gdbt.fit(x_selected, y_train)
gdbt_pred = gdbt.predict(x_selected_test)
var_confusion_matrix = confusion_matrix(y_test, gdbt_pred, labels=None, sample_weight=None)


from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(20,300,20),'tol':range(20,200,20),'max_depth':range(2,5,1)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.03, subsample=0.75),param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)

gsearch1.fit(x_selected,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


########### ({'max_depth': 2, 'n_estimators': 160, 'tol': 60}, 0.8)
gdbt = GradientBoostingClassifier(tol=60, subsample=0.75, n_estimators=160, max_features=x_selected.shape[1], 
                                 max_depth=2, learning_rate=0.01)


# 訓練模型
gdbt.fit(x_selected, y_train)

# 預測測試集
y_pred = gdbt.predict(x_selected_test)
var_confusion_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)


update_y_pred = gdbt.predict_proba(x_selected_update)[:,1]

sub = pd.DataFrame({'name': ids, 'poi': update_y_pred})
sub.to_csv('test1024_gdbt.csv', index=False)