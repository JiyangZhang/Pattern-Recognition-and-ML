import irisData
import Information_Gain
from collections import Counter

dataset = irisData.depose()  # a list of class iris dataset with attributes list and class name

class DecisionTree():
    '''
    define the DecisionTree class, the item of it is a list of dataset
    '''
    def __init__(self, dataset, attr_list): # a list of dataclass object, a list of kinds of classes
        self.key=dataset
        self.atr_list = attr_list
        self.cls = None
        self.leftChild=None
        self.rightChild=None


def TreeGenerate(dataset, attr_list, class_value): # dataset is a list of class attr,class_name
    DT = DecisionTree(dataset, attr_list)
    # CASE I: all the elements in the dataset are from the same class
    if [True for i in range(len(DT.key))] == [i.cls == DT.key[0].cls for i in DT.key]:
        DT.cls = DT.key[0].cls
        return DT
    # CASE II: attr_value is none
    elif DT.atr_list == []:
        cls = []
        for i in DT.key:
            cls.append(i.cls)
        DT.cls = Counter(cls).most_common(1)[0][0]
        return DT
    # CASE III
    else:
        d = Information_Gain.get_split(DT.key, attr_list)
        left = d['groups'][0]
        attr_list_new = attr_list
        attr_list_new.remove(d['index'])
        DT.leftChild = TreeGenerate(left, attr_list_new, class_value)
        right = d['groups'][1]
        DT.rightChild = TreeGenerate(right, attr_list_new, class_value)
    return DT


if __name__ == '__main__':
    TreeGenerate(dataset, [0,1,2,3], [0,1,2])

