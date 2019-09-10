# -*- coding: utf-8 -*-
"""
https://www.pypandas.cn/docs/
https://medium.com/datainpoint/%E5%BE%9E-pandas-%E9%96%8B%E5%A7%8B-python-%E8%88%87%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8%E4%B9%8B%E6%97%85-8dee36796d4a
"""
import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = './csv/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

# 這個 DataFrame 有幾列有幾欄
app_train.shape
#這個 DataFrame 的變數資訊
app_train.columns
#這個 DataFrame 的列索引資訊
app_train.index
# 關於 DataFrame 的詳細資訊
app_train.info()
#關於 DataFrame 各數值變數的描述統計
app_train.describe()

"""
dplyr 的基本功能是六個能與 SQL 查詢語法相互呼應的函數：

    filter() 函數：SQL 查詢中的 where 描述
    select() 函數：SQL 查詢中的 select 描述
    mutate() 函數：SQL 查詢中的衍生欄位描述
    arrange() 函數：SQL 查詢中的 order by 描述
    summarise() 函數：SQL 查詢中的聚合函數描述
    group_by() 函數：SQL 查詢中的 group by 描述
"""

app_train.head(100)


app_train[app_train['SK_ID_CURR']==100002]








