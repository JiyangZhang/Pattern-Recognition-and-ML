import numpy as np
import dataset as ds
import copy

def initialize(X, dim, number):  # dim is the number of the features of X
    w = np.random.randn(dim, 1)
    b = np.array([[0]]).T
    a = np.append(w, b, axis = 0)
    assert(a.shape == (dim+1, 1))
    z = np.append(X, np.ones((1,number)), axis = 0)
    #assert(z.shape == (dim+1,1))
    return a, z


def adjust(z, Y, dim):
    Y_new = Y.repeat(dim + 1, axis = 0)
    z[(Y_new == 0)] *= -1
    return z


def optimize(n, a, z, number):  # n: learning speed  number: the # of training data
    flag = 1
    time = 0
    while flag == -1 or time == 0:
        flag = 2
        time += 1
        for j in range(number):
            z_plus = z[:,j].reshape(3,1)
            if np.dot(a.T, z_plus) <= 0:
                a = a + n * z_plus
                flag = -1
    return a


def model(X, Y, n):  # X: the data set Y: label set
    dim, number = X.shape
    a, z = initialize(X, dim, number)
    z_new = adjust(z, Y, dim)
    a_new = optimize(n, a, z_new, number)
    return a_new, z


def accuracy(Y, y):
    dim = Y.shape[1]
    one = np.ones((1, dim))
    result = np.sum(one[Y == y])
    ac = result/dim * 100
    return ac

if __name__ == '__main__':
# create the data set
    X, Y, w = ds.mk_data(120)
    Y_nature = copy.deepcopy(Y)
    Y[Y<0] = 0
    number = X.shape[1]
# training part
    train = X[:,0:100].reshape(2,100)
    label = Y[:,0:100].reshape(1,100)
    a_new, z = model(train, label, 0.02)
# testing part
    test = X[:,100:120].reshape(2,20)
    result = np.dot(a_new.T, np.append(test, np.ones((1,20)), axis = 0))
    result[result > 0] = 1
    result[result < 0] = 0
    print('Accuracy: %d' % accuracy(result, Y[:, 100:120]) + "%")
    x = np.arange(-1.1, 1.1, .01)
    y = np.arange(-1.1, 1.1, .01)
    x, y = np.meshgrid(x, y)
    f = a_new[0] * x + a_new[1] * y + a_new[2]
    pl = ds.plot(X, Y_nature)
    pl.contour(x, y, f, 0, colors = 'black', linewidth = 0.001)
    pl.show()







