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


var_train_columns = ['bonus', 'deferral_payments', 'deferred_income',
       'exercised_stock_options', 'long_term_incentive', 'other' ,
       'restricted_stock', 'salary', 'total_payments', 'total_stock_value']

df_cp = df_train_tmp[var_train_columns]
    
df_cp['other'] = df_cp['other'].clip(0, 0.3*(10**7))
df_cp['other'] = df_cp['other'].fillna(0)
df_cp['total_payments'] = df_cp['total_payments'].clip(0, 0.2*(10**8))
df_cp['total_payments'] = df_cp['total_payments'].fillna(0)

df_cp['deferral_payments'] = df_cp['deferral_payments'].fillna(0)
df_cp['deferred_income'] = df_cp['deferred_income'].fillna(0)
df_cp['long_term_incentive'] = df_cp['long_term_incentive'].fillna(0)
df_cp['bonus'] = df_cp['bonus'].fillna(0)
df_cp['exercised_stock_options'] = df_cp['exercised_stock_options'].fillna(0)
df_cp['restricted_stock'] = df_cp['restricted_stock'].fillna(0)
df_cp['total_stock_value'] = df_cp['total_stock_value'].fillna(0)

df_cp['salary'] = df_cp['salary'].fillna(df_cp['salary'].median())


na_check(df_cp)

feat_labels = df_cp.columns
feat_labels
########## 前處理理 (Processing) end


# 將資料最大最小化
#df_cp = MinMaxScaler().fit_transform(df_cp)  
df_cp = df_cp.as_matrix()
# 將前述轉換完畢資料 df , 重新切成 train_X, test_X
#train_num = train_Y.shape[0]
#train_X = df_cp[:train_num]

#update_test_X = df_cp[train_num:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_cp, train_Y, test_size=0.25, random_state=4)


########## model start
from sklearn.ensemble import GradientBoostingClassifier
gdbt = GradientBoostingClassifier(learning_rate=0.01)


# 訓練模型
gdbt.fit(x_train, y_train)

# 預測測試集
y_pred = gdbt.predict(x_test)

y_pred_proba =  gdbt.predict_proba(x_test)[:,1]


########## model end



########## 糢型憑估  start
from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

check_view = pd.DataFrame({'pred_poi': y_pred_proba,'poi':y_test})
check_view = check_view.sort_values(by=['pred_poi'])

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
#x_selected_update = update_test_X[:, importances > threshold]

gdbt = GradientBoostingClassifier(tol=180, subsample=0.75, n_estimators=160, max_features=x_selected.shape[1], 
                                 max_depth=4, learning_rate=0.01)

gdbt.fit(x_selected, y_train)
gdbt_pred = gdbt.predict(x_selected_test)
var_confusion_matrix = confusion_matrix(y_test, gdbt_pred, labels=None, sample_weight=None)

acc = accuracy_score(y_test, gdbt_pred)
print("Accuracy: ", acc)



from sklearn.model_selection import GridSearchCV
param_test1 = {'n_estimators':range(20,300,20),'tol':range(20,200,20),'max_depth':range(2,5,1)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.01, subsample=0.75),param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)

gsearch1.fit(x_selected,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


########### ({'max_depth': 2, 'n_estimators': 160, 'tol': 60}, 0.8)
gdbt = GradientBoostingClassifier(tol=180, subsample=0.75, n_estimators=160, max_features=x_selected.shape[1], 
                                 max_depth=4, learning_rate=0.01)


# 訓練模型
gdbt.fit(x_selected, y_train)

# 預測測試集
y_pred = gdbt.predict(x_selected_test)
var_confusion_matrix = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)


update_y_pred = gdbt.predict_proba(x_selected_update)[:,1]

sub = pd.DataFrame({'name': ids, 'poi': update_y_pred})
sub.to_csv('test1024_gdbt.csv', index=False)




df_cp = df_test[var_train_columns]
    
df_cp['other'] = df_cp['other'].clip(0, 0.3*(10**7))
df_cp['other'] = df_cp['other'].fillna(0)
df_cp['total_payments'] = df_cp['total_payments'].clip(0, 0.2*(10**8))
df_cp['total_payments'] = df_cp['total_payments'].fillna(0)

df_cp['deferral_payments'] = df_cp['deferral_payments'].fillna(0)
df_cp['deferred_income'] = df_cp['deferred_income'].fillna(0)
df_cp['long_term_incentive'] = df_cp['long_term_incentive'].fillna(0)
df_cp['bonus'] = df_cp['bonus'].fillna(0)
df_cp['exercised_stock_options'] = df_cp['exercised_stock_options'].fillna(0)
df_cp['restricted_stock'] = df_cp['restricted_stock'].fillna(0)
df_cp['total_stock_value'] = df_cp['total_stock_value'].fillna(0)

df_cp['salary'] = df_cp['salary'].fillna(df_cp['salary'].median())

var_key = ['long_term_incentive','other','bonus','exercised_stock_options']
update_test_x = df_cp[var_key]

gdbt_pred = gdbt.predict_proba(update_test_x)[:,1]

sub = pd.DataFrame({'name': ids, 'poi': gdbt_pred})
sub.to_csv('test1025-2_gdbt.csv', index=False)




#######
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_selected,y_train)
rf_pred = rf.predict(x_selected_test)
var_confusion_matrix = confusion_matrix(y_test, rf_pred, labels=None, sample_weight=None)

acc = accuracy_score(y_test, rf_pred)
print("Accuracy: ", acc)

rf_pred = rf.predict_proba(update_test_x)[:,1]

sub = pd.DataFrame({'name': ids, 'poi': rf_pred})
sub.to_csv('test1025_rf.csv', index=False)
