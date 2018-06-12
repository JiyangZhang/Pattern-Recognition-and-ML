from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
digits = datasets.load_digits()


def dispose():
    training_set_data = np.array(iris.data[:130]).T
    test_set_data = np.array(iris.data[130:]).T
    training_set_label = np.zeros((3, 130))
    test_set_label = np.zeros((3, 20))
    for i in range(130):
        training_set_label[iris.target[i], i] = 1
    for i in range(20):
        test_set_label[iris.target[i+130], i] = 1
    return training_set_data, training_set_label, test_set_data, test_set_label


if __name__ == '__main__':
    X, Y, x, y = dispose()
    print(Y.shape)