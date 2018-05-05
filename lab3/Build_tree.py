import Information_Gain
import RirsData
dataset = RirsData.depose()


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']   # a dictionary write down the
    del(node['groups'])
# check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
# check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
# process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = Information_Gain.get_split(left, attrs=[0,1,2,3])
        split(node['left'], max_depth, min_size, depth+1)
# process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = Information_Gain.get_split(right, attrs=[0,1,2,3])
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = Information_Gain.get_split(dataset, attrs=[0,1,2,3])
    split(root, max_depth, min_size, 1)
    return root
