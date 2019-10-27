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

data_path = './'
rf = pd.read_csv(data_path + 'test1023_rf.csv')
gdbt = pd.read_csv(data_path + 'test1025-2_gdbt.csv')


ids = gdbt['name']
test = gdbt.loc[(gdbt['name']=='BELDEN TIMOTHY N')]



var_c = pd.merge(gdbt, rf, how='inner', on='name', left_on=None, right_on=None,
      left_index=False, right_index=False, sort=True,
      suffixes=('_x', '_y'), copy=True, indicator=False)


var_c['poi'] = var_c['poi_x']*0.6 + var_c['poi_y']*0.4
var_c[['name','poi']]

var_c[['name','poi']].to_csv('test1027_stocking.csv', index=False)