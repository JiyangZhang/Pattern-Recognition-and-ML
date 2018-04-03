import numpy as np
import random
import pylab as pl
# create the data set

# AX=0 相当于matlab中 null(a','r')
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol * s[0]).sum()
    return rank, v[rank:].T.copy()


# 符号函数，之后要进行向量化
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1

# noisy=False，那么就会生成N的dim维的线性可分数据X，标签为y
# noisy=True, 那么生成的数据是线性不可分的,标签为y


def mk_data(N, noisy=False):
    rang = [-1, 1]
    dim = 2

    X = np.random.rand(dim, N) * (rang[1] - rang[0]) + rang[0]

    while True:
        Xsample = np.concatenate((np.ones((1, dim)), np.random.rand(dim, dim) * (rang[1] - rang[0]) + rang[0]))
        k, w = null(Xsample.T)
        y = np.vectorize(sign)(np.dot(w.T, np.concatenate((np.ones((1, N)), X))))
        if np.all(y):
            break

    if noisy == True:
        idx = random.sample(range(1, N), N / 10)
        y[idx] = -y[idx]

    return (X, y, w)


def plot(X, y): # num : the # of created data
    y = y.repeat(X.shape[0], axis=0)
    x1 = X[y == 1]
    s = x1.size
    x1 = x1.reshape(2, int(s / 2))
    x2 = X[y == -1]
    s = x2.size
    x2 = x2.reshape(2, int(s / 2))
    pl.plot(x1[0, :], x1[1, :], 'g*')  # 调用pylab的plot函数绘制曲线
    pl.plot(x2[0, :], x2[1, :], 'r*')
    return pl

if __name__ == '__main__':
    X, y, w = mk_data(100)
    a = plot(X, y)
    a.show()





