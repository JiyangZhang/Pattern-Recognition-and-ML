from sklearn import datasets
import sklearn
import numpy as np

def data_process():
    cancers = datasets.load_breast_cancer(return_X_y=True)
    cancers_std = sklearn.preprocessing.scale(cancers[0])
    cancers_nom = sklearn.preprocessing.normalize(cancers_std)
    training_set = (cancers_nom[0:455].T, cancers[1][0:455].reshape(1, 455))
    test_set = (cancers_nom[455:569].T, cancers[1][455:569].reshape(1, 114))
    return training_set, test_set





