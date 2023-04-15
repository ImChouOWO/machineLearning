import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


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
    print("feature",feature)
    # 將資料以7:3的方式切割
    X_train, X_test, y_train, y_test = train_test_split(data[feature], data[target], test_size=0.3, random_state=42,stratify=data[target])
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    if X_train.shape[1] != 77 or X_test.shape[1] != 77:
        print("Error: Feature dimension is incorrect")
    pca = PCA(n_components=58)
    X_train_extracted = pca.fit_transform(X_train)
    X_test_extracted = pca.transform(X_test)

    # 使用新特征来训练和测试模型
    logisticRegression(X_train_extracted, X_test_extracted, y_train, y_test)

    return X_train, X_test, y_train, y_test

    
def logisticRegression(X_train, X_test, y_train, y_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    #使用訓練數據拟合模型
 
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
    lr.fit(X_train_std, y_train)
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    #计算模型的准确性
    print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('images/03_06.png', dpi=300)
    plt.show()

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = np.random.rand(45, 3)
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()] + [np.zeros(xx1.ravel().shape[0])]*56).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c=colors,
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')





    



if __name__== "__main__":
    data = pd.read_csv("code/online_shoppers_intention.csv")
    data =  missingData(data)
    data = dummyTransformation(data)
    dataPreprocessing = dataCut(data)
    
    
    



