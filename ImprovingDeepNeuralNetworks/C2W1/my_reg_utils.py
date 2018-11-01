#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:09:54 2018

@author: dingqingpeng

Description:
    Functions used in implementing regularization
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def relu(x):
    """
    Compute the relu of x

    Arguments:
        x -- A scalar or numpy array of any size.

    Return:
        s -- relu(x)
    """
    s = np.maximum(0,x)
    return s

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
        x -- A scalar or numpy array of any size.

    Return:
        s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_2D_dataset(plot = False):
    """
    Load 2 dimentional dataset
    
    Arguments:
        plot -- If True, draw scatter of train set
    
    Returns:
        train_X, train_Y, test_X, test_Y -- train set and test set
    """
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    
    if plot:
        plt.scatter(train_X[0, :], train_X[1, :], c = train_Y.reshape(-1,), s = 40, cmap = plt.cm.Spectral)
        plt.show()
    return train_X, train_Y, test_X, test_Y

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss).
    
    Arguments:
        X -- input dataset, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape ()
                        b1 -- bias vector of shape ()
                        W2 -- weight matrix of shape ()
                        b2 -- bias vector of shape ()
                        W3 -- weight matrix of shape ()
                        b3 -- bias vector of shape ()
    
    Returns:
        A3 -- last activation value, output of the forward propagation, of shape (1,1)
        cache -- tuple, information stored for computing the backward propagation
    """
    # Retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def forward_propagation_with_dropout(X, parameters, keep_prob):
    """
    Implements the forward propagation (and computes the loss).
    
    Arguments:
        X -- input dataset, of shape (input size, number of examples)
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape ()
                        b1 -- bias vector of shape ()
                        W2 -- weight matrix of shape ()
                        b2 -- bias vector of shape ()
                        W3 -- weight matrix of shape ()
                        b3 -- bias vector of shape ()
        keep_prob -- probability of keeping a neuron active during drop-out, scalar
    
    Returns:
        A3 -- last activation value, output of the forward propagation, of shape (1,1)
        cache -- tuple, information stored for computing the backward propagation
    """
    
    np.random.seed(1)
    
    # Retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    # Carry out dropout
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = (D1 < keep_prob)
    A1 *= D1
    A1 /= keep_prob
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2= np.random.rand(A2.shape[0], A2.shape[1])
    D2 = (D2 < keep_prob)
    A2 *= D2
    A2 /= keep_prob
    
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache

def compute_cost(An, Y):
    """
    Implement the cost function
    
    Arguments:
        An -- post-activation, output of forward propagation
        Y -- "true" labels vector, same shape as An
    
    Returns:
        cost - value of the cost function
    """
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(An)) + np.multiply(1 - Y, np.log(1 - An))
    cost = -1./m * np.nansum(logprobs)
    return cost

def compute_cost_with_regularization(An, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization
    
    Arguments:
        An -- post-activation, output of forward propagation
        Y -- "true" labels vector, same shape as An
        parameters -- python dictionary containing parameters of the model
        lambd -- regularization hyperparameter, scalar
    
    Returns:
        cost - value of the cost function
    """
    m = Y.shape[1]
    
    # Retrieve W parameter
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    # cross-entropy part of the cost
    cross_entropy_cost = compute_cost(An, Y)
    
    # L2 regularization part of the cost
    L2_regularization_cost = 1./m * lambd/2. * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation.
    
    Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
        cache -- cache output from forward_propagation()
    
    Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis = 1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, A2 > 0)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis = 1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 > 0)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis = 1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularization.
    
    Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar
    
    Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd / m * W3
    db3 = 1./m * np.sum(dZ3, axis = 1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, A2 > 0)
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1./m * np.sum(dZ2, axis = 1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, A1 > 0)
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1./m * np.sum(dZ1, axis = 1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients
    
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.
    
    Arguments:
        X -- input dataset, of shape (2, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation_with_dropout()
        keep_prob -- probability of keeping a neuron active during drop-out, scalar
    
    Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis = 1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dA2 *= D2
    dA2 /= keep_prob
    dZ2 = np.multiply(dA2, A2 > 0)
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis = 1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dA1 *= D1
    dA1 /= keep_prob
    dZ1 = np.multiply(dA1, A1 > 0)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis = 1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(i)] = Wi
                        parameters['b' + str(i)] = bi
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(i)] = dWi
                        grads['db' + str(i)] = dbi
        learning_rate -- the learning rate, scalar.
    
    Returns:
        parameters -- python dictionary containing your updated parameters 
    """
    
    for k in range(1, 1 + len(parameters) // 2):
        parameters["W" + str(k)] -= learning_rate * grads["dW" + str(k)]
        parameters["b" + str(k)] -= learning_rate * grads["db" + str(k)]
    return parameters

def predict(X, y, parameters, print_results = False):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples you would like to label
        y -- true "label" vector (containing 0s and 1s)
        parameters -- parameters of the trained model
    
    Returns:
        p -- predictions for the given dataset X
    """
    
    p = np.zeros((1, X.shape[1]), dtype = np.int)
    
    # Forward propagation
    a3, cache = forward_propagation(X, parameters)
    
    # convert probs to 0/1 predictions
    p = np.uint8(a3 > 0.5)
    
    if print_results:
        print("Accuracy: " + str(np.mean(p[0, :] == y[0, :])))
    
    return p

def plot_decision_boundary(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
        X -- data set of examples you would like to label
        y -- true "label" vector (containing 0s and 1s)
    
    Returns:
        None
    """
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, y, parameters)
    
    # Plot the contour and training examples
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap = plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c = y.reshape(-1,), cmap = plt.cm.Spectral)
    plt.show()
    
    return None





