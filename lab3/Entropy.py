import math
from collections import defaultdict
def Entropy(class_value, attr):
    """
    # Calculate the entropy for a split dataset
    :param class_value: a list of kinds of classes
    :param attr: a list of classes
    :return: a number entropy
    """
    entropy = 0
    class_num = len(class_value)  # the number of different classes
    dic = defaultdict(int)
    for i in attr:
        dic[i] += 1
    for k in dic.keys():
        if dic[k] == 0:
            continue
        p = dic[k]/len(attr)
        entropy += -1 * p * math.log(p,2)
    return entropy

def IG(dataset, dataright, dataleft, class_value):
    '''
    # calculate the information gain of a certain attribute
    :param dataset: the input whole dataset
    :param dataright: the right dataset split according to an attribute
    :param dataleft: the left part dataset split according to an attribute
    :param class_value: nunber of kinds of classes
    :return:
    '''
    entropyD = Entropy(class_value, dataset)  # the param dataset e.g [0,1,2,0...]
    entropyDr = Entropy(class_value, dataright)
    entropyDl = Entropy(class_value, dataleft)
    infoGain = entropyD - ((len(dataleft)/len(dataset))*entropyDl + (len(dataright)/len(dataset))*entropyDr)
    return infoGain




