import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 讀取資料集
df = pd.read_csv('online_shoppers_intention.csv')
print(df)

# 1. 處理 missing data 的特徵與數量
# 檢查 missing data 的總數量
print(df.isnull().sum())

# 處理 delete 與填補空值
# 刪除 missing data 過多的特徵
df = df.drop(['Administrative', 'Informational', 'ProductRelated'], axis=1)
# 針對某些特徵，可以填補空值
df['Month'] = df['Month'].fillna(df['Month'].mode()[0])

# 2+3. 將所有類別特徵轉換成 dummy
Month_mapping = {label: idx for idx, label in enumerate(
    np.unique(df['Month']))}  # 做label
df['Month'] = df['Month'].map(Month_mapping)  # 把原本的值替換成label
print(Month_mapping)

VisitorType_mapping = {label: idx for idx,
                       label in enumerate(np.unique(df['VisitorType']))}
df['VisitorType'] = df['VisitorType'].map(VisitorType_mapping)
print(VisitorType_mapping)

dummies = pd.get_dummies(df)  # 轉型態


# 4. 分割成 train 與 test data set
# 將資料集分割成訓練集和測試集
X = df.drop(['Revenue'], axis=1)
y = df['Revenue']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
print(X_train)
print(X_test)

# 5-1. 特徵縮放: MinMax 與 standardized
# 將數值特徵進行 MinMax 縮放
scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(
    X_train[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']])
X_test_minmax = scaler.transform(
    X_test[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']])

# 將數值特徵進行標準化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(
    X_train[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']])
X_test_std = scaler.transform(
    X_test[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']])

print(X_train_minmax)
print(X_test_minmax)

print(X_train_std)
print(X_test_std)

# 5-2.
describe = df.describe()  # 顯示統計數值
print(describe)
