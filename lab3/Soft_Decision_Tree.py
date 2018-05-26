from numpy import np
def sigmoid(x):
    def sigmoid(x):
        s = 1 / (1 + np.exp(x * -1))
        return s

class Node():
    def __init__(self, dim):
        self.isLeaf = True
        self.parent = None
        self.left = None
        self.right = None
        self.w = np.zeros(1,dim)
        self.w0 = 0

def learn(X, dim):
    # suppose the dataset is X
    for i in range(len(X)):
        t = 1
        m = Node(dim)
        while m.parent != None:
            p = m.parent
            if m.left:
               t *= sigmoid(np.dot(p.w, X[i])+ p.w0)
            else:
                t *= (1-sigmoid(np.dot(p.w, X[i]) + p.w0))
            m = m.parent
