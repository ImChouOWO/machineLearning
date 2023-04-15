import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def missingData(data):

    print("DataFrame 中是否存在遺失值:",data.isnull().values.any(),"\n")
    print("每個欄位的遺失值數量:\n",data.isnull().sum())
    data.dropna(inplace=True)
    return data

def dummyTransformation(data):
    print("欄位類型：\n",data.dtypes,"\n")
    
    # 將以下類別尺度的欄位做轉換
    colEncodin =['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
    data =  pd.get_dummies(data, columns=colEncodin)

    print("one-hot-encoding：\n",data)
    return data
def dataCut(data):
    # 目標欄位Revenue_True，用於預測顧客是否購買
    target =["Revenue_True"]
    feature = [col for col in data.columns if col != target]
    # 將資料以7:3的方式切割
    train_data, test_data, train_target, test_target = train_test_split(data[feature], data[target], test_size=0.3, random_state=42)
    returnData = featureScaling(train_data,test_data)
    return [returnData[0],returnData[1],train_target,test_target]

def featureScaling(train_data,test_data):

    # 創建 StandardScaler 物件
    scaler = StandardScaler()

    # 對 train_data 進行特徵縮放
    
    train_data_scaled = scaler.fit_transform(train_data)
    print("train_data 特徵縮放 \n",train_data_scaled)

    # 對 test_data 進行特徵縮放
    test_data_scaled = scaler.transform(test_data)
    print("test_data 特徵縮放 \n",test_data_scaled)
    return [train_data_scaled,test_data_scaled]
    
def logisticRegression(train_data_scaled,test_data_scaled,train_target,test_target):
    # #使用訓練數據拟合模型
    # logreg = LogisticRegression()
    # logreg.fit(train_data_scaled, np.ravel(train_target))


    # #使用測試數據进行模型预测
    # predict = logreg.predict(test_data_scaled)

    # #計算準確性：
    # accuracy = accuracy_score(test_target, predict)

    # print("準確性：", accuracy)


    



if __name__== "__main__":
    data = pd.read_csv("code/online_shoppers_intention.csv")
    data =  missingData(data)
    data = dummyTransformation(data)
    dataPreprocessing = dataCut(data)
    print("final")
    print(dataPreprocessing)
    logisticRegression(dataPreprocessing[0],dataPreprocessing[1],dataPreprocessing[2],dataPreprocessing[3])




