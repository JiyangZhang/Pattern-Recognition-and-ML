'''
calculate information information gain of attrs of each attribute
author : Jiyang
'''
import math
def Entropy(class_value, attr):
    """
    # Calculate the entropy index for a split dataset
    :param class_value: a list of classes
    :param attr:
    :return: entropy
    """
    entropy = 0
    class_num = len(class_value)
    dic = {}
    for i in range(class_num):
        dic[i] = 0
    for i in attr:
        dic[i] += 1
    for k in dic:
        if dic[k] == 0:
            continue
        p = dic[k]/len(attr)
        entropy += -1 * p * math.log(p,2)
    return entropy

def IG(dataset, dataright, dataleft, class_value):  #
    entropyD = Entropy(class_value, dataset)
    entropyDr = Entropy(class_value, dataright)
    entropyDl = Entropy(class_value, dataleft)
    infoGain = entropyD - ((len(dataleft)/len(dataset))*entropyDl + (len(dataright)/len(dataset))*entropyDr)
    return infoGain




