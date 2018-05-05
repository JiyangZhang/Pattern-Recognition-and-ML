import Entropy
import RirsData
def test_split(dataset, threshold, attr_num):   # split the dataset according to the threshold
    left = []     # left smaller, right bigger
    right = []
    for i in dataset:
        if i.attr[attr_num] < threshold:
            left.append(i)
        else:
            right.append(i)
    return left, right

# Select the best split point for a dataset
def get_split(dataset, attrs):  #attrs = [0,1,2,3] in iris
    b_index, b_value, b_score, b_groups = 0, 0, 0, None
    attr_list = []
    for i in dataset:
        attr_list.append(i.cls)
    for i in range(len(attrs)):
        for j in dataset:
            left, right = test_split(dataset, j.attr[i], i)  # left and right is a list of class member
            attr_list1 = []
            attr_list2 = []
            for k in right:
                attr_list1.append(k.cls)
            for p in left:
                attr_list2.append(p.cls)
            infoGain = Entropy.IG(attr_list, attr_list1, attr_list2, [0,1,2])
            if infoGain > b_score:
                b_index, b_value, b_score, b_groups = i, j.attr[i], infoGain, (left, right)
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

if __name__ == '__main__':
    dataset = RirsData.depose()
    print(get_split(dataset, [0,1,2,3]))




