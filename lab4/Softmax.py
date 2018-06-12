import numpy as np
import matplotlib.pyplot as plt
import IrisData

training_set_data, training_set_label, test_set_data, test_set_label = IrisData.dispose()

def softmax(Z):
    cache = Z
    numerator = np.exp(Z)
    denominator = np.sum(numerator, axis=0)
    return numerator/denominator, cache


def softmax_backward(Y, cache):
    Z = cache
    AL, m = softmax(Z)
    dZ = -Y + AL
    return dZ

def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ



def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache  #


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)  # every cache of linear_activation_forward(), tuples of (linear_cache for  (A, W, b), and *)

    #assert (AL.shape == (1, X.shape[1]))

    return AL, caches

def Scost(AL, Y, lamda, w):
    m = AL.shape[1]
    assert(m == 130)
    cost = -1/m *(np.sum(Y * np.log(AL))) + lamda/2 * 1/m * np.sum(w**2)
    return cost

def linear_backward(dZ, cache, lamda=0):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T) + 1/m * lamda * W
    db = 1 / m * np.sum(dZ, axis=1)
    db = db.reshape(b.shape)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lamda):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,lamda)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, lamda):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Lth layer (SOFTMAX -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = linear_activation_backward(Y, caches[L - 1], "softmax", lamda)
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache[0], current_cache[1], \
                                                                           current_cache[2]

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = linear_activation_backward(grads["dA" + str(l + 1)], caches[l], "relu", lamda)
        dA_prev_temp, dW_temp, db_temp = current_cache[0], current_cache[1], current_cache[2]
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.075, num_iterations=3000, print_cost=True, lamda=0.02):  # lr was 0.009
    costs = []  # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = Scost(AL, Y, lamda, parameters['W' + str(len(layers_dims)-1)])

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lamda)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

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

def predict(X, Y, parameters):
    num = X.shape[1]
    L = len(parameters) // 2
    A0 = X
    for l in range(1, L):
        Z = np.dot(parameters["W" + str(l)], A0) + parameters["b" + str(l)]
        A,cache = relu(Z)
        A0 = A
    Z = np.dot(parameters["W" + str(L)], A0) + parameters["b" + str(L)]
    A,cache = softmax(Z)

    maxm = np.max(A, axis=0)
    maxm = maxm * np.ones(A.shape)
    result = np.ones(A.shape)
    result[A<maxm] = 0


    print("accuracy:{}%".format(100 - np.mean(np.abs(result - Y)) * 100))
    return result


if __name__ == '__main__':
    parameters = L_layer_model(training_set_data, training_set_label, [4,6,3], learning_rate = 0.075, num_iterations = 4000)
    print(predict(test_set_data, test_set_label, parameters))