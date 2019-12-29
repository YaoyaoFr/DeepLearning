import numpy as np
import xlrd

def norm(X):
    sample_size = np.size(X, 0)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std

def ridge_regression(k,X,y):
    n = np.size(X,1)

    yX = np.concatenate((y, X), axis=1)
    x = np.corrcoef(yX.T)


    inv = np.linalg.inv(x[1:, 1:] + k*np.eye(n))
    beta = np.matmul(inv, x[0, 1:])
    return beta.T

def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    X = np.zeros(shape=[nrows-1,ncols-1])
    y = np.zeros(shape=[nrows-1,1])

    for row in np.arange(1,nrows):
        row_values = table.row_values(row)
        y[row-1,0]=row_values[0]
        X[row-1,:] = row_values[1:]
        pass

    return X,y

X,y = excel2matrix(path='F:/test.xls')
for k in np.arange(start=0.05, stop=1.05, step=0.05):
    print('k:{:f}, result:{:}'.format(k, ridge_regression(k=k,X=X,y=y)))
# print('k:{:f}, result:{:}'.format(0.99, ridge_regression(k=0.99,X=X,y=y)))