import irisData
import Information_Gain
from collections import Counter
import copy

dataset = irisData.depose()  # a list of class iris dataset with attributes list and class name

class DecisionTree():
    '''
    define the DecisionTree class, the item of it is a list of dataset
    In this file the different attributes can be used for more than one time, there is a depth restriction;
    '''
    def __init__(self, dataset, attr_list, depth): # a list of dataclass object, a list of kinds of classes
        self.key=dataset
        self.atr_list = attr_list
        self.cls = None
        self.leftChild=None
        self.rightChild=None
        self.depth = depth


def TreeGenerate(dataset, attr_list, class_value, depth): # dataset is a list of class attr,class_name
    DT = DecisionTree(dataset, attr_list, depth+1)
    # CASE I: all the elements in the dataset are from the same class
    if [True for i in range(len(DT.key))] == [i.cls == DT.key[0].cls for i in DT.key]:
        DT.cls = DT.key[0].cls
        return DT
    # CASE II: the depth is greater than the depth threshold
    elif DT.depth >5:
        cls = []
        for i in DT.key:
            cls.append(i.cls)
        DT.cls = Counter(cls).most_common(1)[0][0]
        return DT
    # CASE III
    else:
        d = Information_Gain.get_split(DT.key, attr_list)
        left = d['groups'][0]
        attr_list_newl = copy.deepcopy(attr_list)
        #attr_list_newl.remove(d['index'])
        DT.leftChild = TreeGenerate(left, attr_list_newl, class_value, DT.depth)
        right = d['groups'][1]
        attr_list_newr = copy.deepcopy(attr_list)
        #attr_list_newr.remove(d['index'])
        DT.rightChild = TreeGenerate(right, attr_list_newr, class_value, DT.depth)
    return DT


if __name__ == '__main__':
    TreeGenerate(dataset, [0,1,2,3], [0,1,2], -1)  # the depth of the root is -1

