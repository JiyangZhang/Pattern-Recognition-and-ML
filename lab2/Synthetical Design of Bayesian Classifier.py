import math
import matplotlib.pyplot as plt
import numpy as np



class w:  # define the class, its attributes: sigma and data
    def __init__(self, data, prob):
        self.mu = 0
        self.sigma = 0
        self.data = data  # a list of data
        self.priProb = prob  # prior probability
        bar = 0
        for i in self.data:
            self.mu += i
        self.mu = self.mu / len(self.data)
        for i in self.data:
            bar += (i - self.mu) ** 2
        self.sigma = math.sqrt(bar / len(self.data))


def condProb(mu, sigma, x):   # input the number and return the conditional prob
    condp = 1/(2.5066*sigma) * math.exp(-0.5*(x-mu)**2/sigma**2)
    return condp

def postProb(x, w1, w2):  # input the number and the class, suppose there are two classes
    num = condProb(w1.mu, w1.sigma, x) * w1.priProb
    den = condProb(w1.mu, w1.sigma, x) * w1.priProb + condProb(w2.mu, w2.sigma, x) * w2.priProb
    return num/den

def classify(x, cls):  # a list of unclassified object, two classes
    result = []
    for i in x:
        risk = []
        risk.append(condProb(cls[0].mu, cls[0].sigma,i) * cls[0].priProb * 6)
        risk.append(condProb(cls[1].mu, cls[1].sigma,i) * cls[1].priProb * 1)
        if min(risk) == risk[0]:
            result.append(2)
        else:
            result.append(1)
    return result


if __name__ == "__main__":
    # this test the case in the guide
    w1 = [-3.9847,-3.5549,-1.2401,-0.9780,-0.7932,-2.8531,-2.7605,-3.7287,\
-3.5414,-2.2692,-3.4549,-3.0752,-3.9934, -0.9780,-1.5799,-1.4885,\
-0.7431,-0.4221,-1.1186,-2.3462,-1.0826,-3.4196,-1.3193,-0.8367,-0.6579,-2.9683]
    w2 = [2.8792, 0.7932,1.1882,3.0682,4.2532,0.3271,0.9846,2.7648,2.6588]
    W1 = w(w1, 0.9)
    W2 = w(w2, 0.1)
    cls = [W1, W2]
    '''draw the line'''
    xnew = np.linspace(-6, 6, 300)  # 300 represents number of points to make between T.min and T.max
    y1 = np.array([condProb(W1.mu, W1.sigma, x) for x in xnew])
    y2 = np.array([condProb(W2.mu, W2.sigma, x) for x in xnew])
    plt.plot(xnew, y1, label = 'class 1', linewidth = 0.5)
    plt.axis([-6, 6, 0, 0.4])
    plt.plot(xnew, y2, label = 'class 2', linewidth = 0.5)
    plt.legend()
    '''
    ynew = np.array([postProb(x, W1, W2) for x in xnew])
    znew = np.array([postProb(x, W2, W1) for x in xnew])
    plt.plot(xnew, ynew, 'r')
    plt.plot(xnew, znew, 'b')
    '''
    plt.show()
