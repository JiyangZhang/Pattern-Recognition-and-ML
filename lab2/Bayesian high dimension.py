from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import math
import numpy as np

#generate 4 datasets
'''return x array of shape[n_samples, n_features]
y array of shape[n_samples]'''
def createData():
    x,y = make_blobs(n_features = 2, centers = 4, n_samples = 200)
    plt.scatter(x[:,0], x[:,1], marker = 'o', c = y, s =45, edgecolor = 'none')
    plt.show()
    cls = [0,1,2,3]
    w0, w1, w2, w3 = [],[],[],[]
    for i in range(len(y)):
        if y[i] == 0:
            w0.append(x[i])
        elif y[i] == 1:
            w1.append(x[i])
        elif y[i] == 2:
            w2.append(x[i])
        else:
            w3.append(x[i])
    return w0,w1,w2,w3
# 4 datasets w0,w1,w2,w3


class w:
    def __init__(self, data, prob):  # suppose data is a list of tuples (x,y)
        self.mu = np.zeros((1, 2))
        self.sigma = np.zeros((2, 2))
        self.data = data
        self.preProb = prob
        mu1 = 0
        mu2 = 0
        for i in self.data:
            mu1 += i[0]
            mu2 += i[1]
        mu1 = mu1 / len(self.data)
        mu2 = mu2 / len(self.data)
        self.mu[0][0], self.mu[0][1] = mu1, mu2
        bar1 = 0
        bar2 = 0
        for i in self.data:
            bar1 += (i[0] - self.mu[0][0]) ** 2
            bar2 += (i[1] - self.mu[0][1]) ** 2

        self.sigma[0][0] = math.sqrt(bar1 / len(self.data))
        self.sigma[1][1] = math.sqrt(bar2 / len(self.data))


def condProb(mu, sigma, x):  # input the 1*2 vector mu, 2*2 array sigma, 1*2 data x;
    sigma = np.mat(sigma)
    condp = 1 / (2 * math.pi * np.linalg.det(sigma) ** 0.5) * math.exp(-0.5 * np.dot(np.dot((x - mu), sigma),\
                                                                                 (x - mu).T))
    return condp

def classify(x, cls):
    result = []
    for i in x:
        risk = []  #"""risk is the risk of choosing 0123"""
        for j in range(4):
            R = 0
            for k in range(4):
                if k != j:
                    R += condProb(cls[k].mu, cls[k].sigma, i) * cls[k].preProb
            risk.append(R)
        cla = risk.index(min(risk))
        result.append(cla)
    return result

if __name__ == "__main__":
    # this test the case in the guide
    W = createData()  # tuples of data from 4 classes
    priorProb = [0.4, 0.2, 0.1, 0.3]  #priorprobability of classes
    cls = []  # a list contains 4 classes
    for i in range(4):
        new = w(W[i], priorProb[i])
        cls.append(new)
    print('the average is:' + str(cls[0].mu) + '\n' + 'the sigma is:' + str(cls[0].sigma))
    print('the average is:' + str(cls[1].mu) + '\n' + 'the sigma is:' + str(cls[1].sigma))
    print('the average is:' + str(cls[2].mu) + '\n' + 'the sigma is:' + str(cls[2].sigma))
    print('the average is:' + str(cls[3].mu) + '\n' + 'the sigma is:' + str(cls[3].sigma))
'''
    x = W[1]
    result = classify(x, cls)
    #print(result)
'''