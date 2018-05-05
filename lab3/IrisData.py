from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

class data():
    def __init__(self, attr, class_name):
        self.attr = attr  # a list of attributes
        self.cls = class_name

def depose():
    dataset = []
    for i in range(len(iris.target)):
        dataset.append(data(iris.data[i], iris.target[i]))
    return dataset


