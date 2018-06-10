import numpy as np
import matplotlib.pyplot as plt
import BreastData

training_set, test_set = BreastData.data_process()
X = training_set[0]

# Part 1  tool functions
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_prime(x):
    a = sigmoid(x) * (1 - sigmoid(x))
    return a


def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


# Part 2  Initialize

def initialize_parameters_deep(hidden_size, visible_size):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}

    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    parameters['W' + str(1)] = np.random.randn(hidden_size, visible_size) * 2 * r -r
    parameters['b' + str(1)] = np.random.randn(hidden_size, 1) * 0.01
    parameters['W' + str(2)] = np.random.randn(visible_size, hidden_size) * 2 * r - r
    parameters['b' + str(2)] = np.random.randn(visible_size, 1) * 0.01

    assert (parameters['W' + str(1)].shape == (hidden_size, visible_size))
    assert (parameters['b' + str(1)].shape == (hidden_size, 1))

    return parameters


# Part 3
def sparse_autoencoder_cost(parameters, sparsity_param, beta, X, learning_rate):
    """
    parameters: the W1, b1, W2, b2
    sparsity_param: the predetermined sparsity
    beta: the parameters of the cost of sparsity
    X: datasets
    learning_rate： the parameters for updating
    return:
    latest parameters and cost
    """

    # Number of training examples
    m = X.shape[1]

    # Forward propagation
    z1 = np.dot(parameters['W' + str(1)], X) + parameters['b' + str(1)]
    a1 = sigmoid(z1)
    z2 = np.dot(parameters['W' + str(2)], a1) + parameters['b' + str(2)]
    h = sigmoid(z2)

    # Sparsity
    rho_hat = np.sum(a1, axis=1) / m
    rho = sparsity_param

    # Cost function
    cost = np.sum((h - X) ** 2) / (2 * m) +\
           beta * np.sum(KL_divergence(rho, rho_hat))

    # Backprop
    grads = {}
    sparsity_delta = - rho / rho_hat + (1 - rho) / (1 - rho_hat)
    sparsity_delta = sparsity_delta.reshape((25,1))
    assert(sparsity_delta.shape == (25, 1))
    dZ2 = -(X - h) * sigmoid_prime(z2)
    dZ1 = (np.dot(parameters['W1'], dZ2) + beta * sparsity_delta) * sigmoid_prime(z1)
    grads["dW1"] = np.dot(dZ1, X.T) / m
    grads["dW2"] = np.dot(dZ2, a1.T) / m
    db = np.sum(dZ1, axis=1) / m
    grads["db1"] = db.reshape(parameters["b1"].shape)
    db = np.sum(dZ2, axis=1) / m
    grads["db2"] = db.reshape(parameters["b2"].shape)

    # update parameters

    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]

    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]

    return parameters, cost


def autoencoder(X, layers_dims, learning_rate=0.075, num_iterations=3000, print_cost=True,
                    sparsity_param=0.05, beta=0.3):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims[1], layers_dims[0])


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        parameters, cost = sparse_autoencoder_cost(parameters, sparsity_param, beta, X, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

if __name__ == '__main__':
    #parameters = L_layer_model(X, Y, [30,5,1], learning_rate = 0.075, num_iterations = 8000)
    parameters = autoencoder(X, [30, 25, 30], learning_rate=0.1, num_iterations=5000, print_cost=True,
                    sparsity_param=0.05, beta=0.3)

