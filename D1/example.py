#!/usr/bin/env python
# coding: utf-8


# import 是 Python 載入套件的基本語法 (類似 C 語言的 include), 後面接要載入的套件
# import AAAAA as BB, 其中 BB 是代稱, 表示除了載入 AAAAA 之外, 之後都可以用 BB 代替 AAAAA 這個名稱
# 常用套件往往有其對應代稱, numpy的代稱是np, pandas的代稱是pd, matplotlib.pyplot的代稱是plt
# numpy 常用於數值/陣列運算, pandas 擅長資料格式的調整, matplotlib 擅長繪圖
import numpy as np
import matplotlib.pyplot as plt

def main():
    int_w = 3;
    double_b = 0.5;
    # np.linspace 是 numpy.linspace 的意思
    # np.linspace(0, 100, 101)是指 0~100 劃分成 101 個刻度(含頭尾), 所也就是 0, 1, 2,...,100 這 101 個數
    # 這時候, x_lin 因為要記錄不只一個數, 因為 np.linspace() 傳回的是一個 Array, 所以 x_lin 就變成 Array 了
    array_x_lin = np.linspace(0, 100, 101);
    # np.random.randn() 就是 numpy.random.randn(), 會隨機傳回標準常態分布的取樣值
    # np.random.randn(101) 表示取樣了101次, 型態是 Array, 所以其他 + 與 * 的部分都是 Array 的加與乘, 一行就計算了101筆資料
    # 所以最後的結果 y, 也是一個長度 101 的 Array
    array_y = (array_x_lin + np.random.randn(101) * 5) * int_w + double_b;
    add_point(array_x_lin,array_y);
    plt.show();
    
    y_hat = array_x_lin * int_w + double_b;
    add_point(array_x_lin,array_y);
    add_line(array_x_lin,y_hat)
    plt.show();
    
    # 呼叫上述函式, 傳回 y(藍點高度)與 y_hat(紅線高度) 的 MAE
    MAE = mean_absolute_error(array_y, y_hat);
    print("The Mean absolute error is %.3f" % (MAE));
    
    mean_square_error(array_y, y_hat);
    
def add_point(array_x_lin,array_y):
    # b. : b 就是 blue, 點(.) 就是最小單位的形狀, 詳細可以查 matplotlib 的官方說明
    plt.plot(array_x_lin, array_y, 'b.', label = 'data points')
    plt.title("Assume we have data points")
    plt.legend(loc = 2);

def add_line(array_x_lin,y_hat):
    # 上面的 'b.' 是藍色點狀, 下面的 'r-' 是紅色線狀, label 是圖示上的名稱
    plt.plot(array_x_lin, y_hat, 'r-', label = 'prediction')
    plt.title("Assume we have data points (And the prediction)")
    plt.legend(loc = 2);
    
# Python 的函數是另一個新手上手的困難點, 由def開頭, 依序是函數名稱 / 輸入值, 冒號(:)結尾
# 最難讓人習慣的是 Python 的函式與條件判斷, 前後都沒有大括弧(其他程式常見), 而是以四格空白縮排來取代
# 以本例來說, mean_absolute_error 這個函數的定義範圍到 return mae 為止, 因為中間都是縮排, 而 """ 是多行註解(井號是單行註解)
# 函數中, sum(), abs(), len() 都是 Python 原有的方法, 因此可以直接呼叫
def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    # MAE : 將兩個陣列相減後, 取絕對值(abs), 再將整個陣列加總成一個數字(sum), 最後除以y的長度(len), 因此稱為"平均絕對誤差"
    mae = sum(abs(y - yp)) / len(y)
    return mae

# https://zhuanlan.zhihu.com/p/37663120
#
def mean_square_error(y, yp):
    from sklearn.metrics import mean_squared_error
    var_mse = sum(np.power(y - yp,2)) / len(y)
    print("The Mean square error is %.3f" % var_mse)
    
    var_sk_mse = mean_squared_error(y,yp)
    print("The Mean square error sk learn is %.3f" % var_sk_mse)

    
    
    
main();

