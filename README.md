# 3rd-ML100Days

## 機器學習概論
1. [資料介紹與評估資料 (申論+程式碼)](https://github.com/escc1122/3rd-ML100Days/tree/master/D1)
挑戰是什麼?動手分析前請三思
1. [機器學習概論 (申論題)](https://github.com/escc1122/3rd-ML100Days/tree/master/D2)
機器學習、深度學習與人工智慧差別是甚麼? 機器學習又有甚麼主題應用?
1. [機器學習-流程與步驟 ( 申論題)](https://github.com/escc1122/3rd-ML100Days/tree/master/D3)
資料前處理 > 訓練/測試集切分 >選定目標與評估基準 > 建立模型 > 調整參數。熟悉整個 ML 的流程
1. [EDA/讀取資料與分析流程](https://github.com/escc1122/3rd-ML100Days/tree/master/D4)
如何讀取資料以及萃取出想要了解的信息

## 資料清理數據前處理
5. [如何新建一個 dataframe? 如何讀取其他資料? (非 csv 的資料)](https://github.com/escc1122/3rd-ML100Days/tree/master/D5)
1 從頭建立一個 dataframe 2. 如何讀取不同形式的資料 (如圖檔、純文字檔、json 等)
5. [EDA: 欄位的資料類型介紹及處理](https://github.com/escc1122/3rd-ML100Days/tree/master/D6)
了解資料在 pandas 中可以表示的類型
5. [特徵類型](https://github.com/escc1122/3rd-ML100Days/tree/master/D7)
特徵工程依照特徵類型，做法不同，大致可分為數值/類別/時間型三類特徵
5. [EDA資料分佈](https://github.com/escc1122/3rd-ML100Days/tree/master/D8)
用統計方式描述資料
5. [EDA: Outlier 及處理](https://github.com/escc1122/3rd-ML100Days/tree/master/D9)
偵測與處理例外數值點：1. 透過常用的偵測方法找到例外 2. 判斷例外是否正常 (推測可能的發生原因)
5. [數值型特徵 - 去除離群值](https://github.com/escc1122/3rd-ML100Days/tree/master/D10)
數值型特徵若出現少量的離群值，則需要去除以保持其餘數據不被影響
5. [常用的數值取代：中位數與分位數連續數值標準化](https://github.com/escc1122/3rd-ML100Days/tree/master/D11)
偵測與處理例外數值 1. 缺值或例外取代 2. 數據標準化
5. [數值型特徵-補缺失值與標準化](https://github.com/escc1122/3rd-ML100Days/tree/master/D12)
數值型特徵首先必須填補缺值與標準化，在此複習並展示對預測結果的差異
5. [DataFrame operationData frame merge/常用的 DataFrame 操作](https://github.com/escc1122/3rd-ML100Days/tree/master/D13)
 1 常見的資料操作方法 2. 資料表串接
5. [程式實作 EDA: correlation/相關係數簡介](https://github.com/escc1122/3rd-ML100Days/tree/master/D14)
 1 了解相關係數 2. 利用相關係數直觀地理解對欄位與預測目標之間的關係
5. [EDA from Correlation](https://github.com/escc1122/3rd-ML100Days/tree/master/D15)
深入了解資料，從 correlation 的結果下手
5. [EDA: 不同數值範圍間的特徵如何檢視/繪圖與樣式Kernel Density Estimation (KDE)](https://github.com/escc1122/3rd-ML100Days/tree/master/D16)
1 如何調整視覺化方式檢視數值範圍 2. 美圖修修 - 轉換繪圖樣式
5. [EDA: 把連續型變數離散化](https://github.com/escc1122/3rd-ML100Days/tree/master/D17)
簡化連續性變數
5. [程式實作 把連續型變數離散化](https://github.com/escc1122/3rd-ML100Days/tree/master/D18)
深入了解資料，從簡化後的離散變數下手
5. [Subplots](https://github.com/escc1122/3rd-ML100Days/tree/master/D19)
探索性資料分析 - 資料視覺化 - 多圖檢視 1. 將數據分組一次呈現 2. 把同一組資料相關的數據一次攤在面前
5. [Heatmap & Grid-plot](https://github.com/escc1122/3rd-ML100Days/tree/master/D20)
探索性資料分析 - 資料視覺化 - 熱像圖 / 格狀圖 1. 熱圖：以直觀的方式檢視變數間的相關性 2. 格圖：繪製變數間的散佈圖及分布
5. [模型初體驗 Logistic Regression](https://github.com/escc1122/3rd-ML100Days/tree/master/D21)
在我們開始使用任何複雜的模型之前，有一個最簡單的模型當作 baseline 是一個好習慣
## 資料科學特徵工程技術

22. [特徵工程簡介](https://github.com/escc1122/3rd-ML100Days/tree/master/D22)
介紹機器學習完整步驟中，特徵工程的位置以及流程架構
22. [數值型特徵 - 去除偏態](https://github.com/escc1122/3rd-ML100Days/tree/master/D23)
數值型特徵若分布明顯偏一邊，則需去除偏態以消除預測的偏差
22. [類別型特徵 - 基礎處理](https://github.com/escc1122/3rd-ML100Days/tree/master/D24)
介紹類別型特徵最基礎的作法 : 標籤編碼與獨熱編碼
22. [類別型特徵 - 均值編碼](https://github.com/escc1122/3rd-ML100Days/tree/master/D25)
類別型特徵最重要的編碼 : 均值編碼，將標籤以目標均值取代
22. [類別型特徵 - 其他進階處理](https://github.com/escc1122/3rd-ML100Days/tree/master/D26)
類別型特徵的其他常見編碼 : 計數編碼對應出現頻率相關的特徵，雜湊編碼對應眾多類別而無法排序的特徵
22. [時間型特徵](https://github.com/escc1122/3rd-ML100Days/tree/master/D27)
時間型特徵可抽取出多個子特徵，或周期化，或取出連續時段內的次數
22. [特徵組合 - 數值與數值組合](https://github.com/escc1122/3rd-ML100Days/tree/master/D28)
特徵組合的基礎 : 以四則運算的各種方式，組合成更具預測力的特徵
22. [特徵組合 - 類別與數值組合](https://github.com/escc1122/3rd-ML100Days/tree/master/D29)
類別型對數值型特徵可以做群聚編碼，與目標均值編碼類似，但用途不同
22. [特徵選擇](https://github.com/escc1122/3rd-ML100Days/tree/master/D30)
介紹常見的幾種特徵篩選方式
22. [特徵評估](https://github.com/escc1122/3rd-ML100Days/tree/master/D31)
介紹並比較兩種重要的特徵評估方式，協助檢測特徵的重要性
22. [分類型特徵優化 - 葉編碼](https://github.com/escc1122/3rd-ML100Days/tree/master/D32)
葉編碼 : 適用於分類問題的樹狀預估模型改良

## 機器學習基礎模型建立
33. [機器如何學習?](https://github.com/escc1122/3rd-ML100Days/tree/master/D33)
了解機器學習的定義，過擬合 (Overfit) 是甚麼，該如何解決
33. [訓練/測試集切分的概念](https://github.com/escc1122/3rd-ML100Days/tree/master/D34)
為何要做訓練/測試集切分？有什麼切分的方法？
33. [regression vs. classification](https://github.com/escc1122/3rd-ML100Days/tree/master/D35)
回歸問題與分類問題的區別？如何定義專案的目標
33. [評估指標選定/evaluation metrics](https://github.com/escc1122/3rd-ML100Days/tree/master/D36)
專案該如何選擇評估指標？常用指標有哪些？
33. [regression model 介紹 - 線性迴歸/羅吉斯回歸](https://github.com/escc1122/3rd-ML100Days/tree/master/D37)
線性迴歸/羅吉斯回歸模型的理論基礎與使用時的注意事項
33. [regression model 程式碼撰寫](https://github.com/escc1122/3rd-ML100Days/tree/master/D38)
如何使用 Scikit-learn 撰寫線性迴歸/羅吉斯回歸模型的程式碼
33. [regression model 介紹 - LASSO 回歸/ Ridge 回歸](https://github.com/escc1122/3rd-ML100Days/tree/master/D39)
LASSO 回歸/ Ridge 回歸的理論基礎與與使用時的注意事項
[參考](https://www.zhihu.com/question/38121173)
33. [11111](https://github.com/escc1122/3rd-ML100Days/tree/master/D31)
33. [11111](https://github.com/escc1122/3rd-ML100Days/tree/master/D31)
