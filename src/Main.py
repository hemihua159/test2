import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_predict

if __name__ == '__main__':
    # 显示所有行
    pd.set_option('display.max_rows', None)

    train_path = os.path.join('~/s3data/dataset', 'train.csv')
    test_path = os.path.join('~/s3data/dataset', 'test.csv')

    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = train_df[['AT','V','AP','RH']]

    y = train_df[['PE']]

    linreg = LinearRegression()
    linreg.fit(X, y)
    y_pred = linreg.predict(X)

    # 检查缺失

    print(train_df.isnull().any())
    np.any(np.isnan(train_df))
    train_df.dropna(inplace=True)
    # 缺失填充
    train_df.fillna('100')
    print("begin")
    print(linreg.intercept_)
    print(linreg.coef_)

    print("MSE:", metrics.mean_squared_error(y, y_pred))
    print ("RMSE:", np.sqrt(metrics.mean_squared_error(y, y_pred)))


    # 交叉验证
    predicted = cross_val_predict(linreg, X, y, cv=10)
    print("MSE:", metrics.mean_squared_error(y, predicted))

    print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))



    #打印图象
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


    path = 'preprocess_data.csv'
    train_df.to_csv(path, index=False)
