#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 18:51:22 2018

@author: dingqingpeng

Description:
    first assignment of "Improving Deep Neural Networks"
    Training your neural network requires specifying an initial value of the weights.
    A well chosen initialization method will help learning.
"""

"""
load the packages and the planar dataset to classify
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

#%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)         #set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
load image dataset: blue/red dots in circles
"""
train_X, train_Y, test_X, test_Y = load_dataset()

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implement a 3-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    
    Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
        learning_rate -- learning rate for gradient descent 
        num_iterations -- number of iterations to run gradient descent
        print_cost -- if True, print the cost every 1000 iterations
        initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
        parameters -- parameters learnt by the model
    """
    grads = {}
    costs = []      #keep track of the loss
    m = X.shape[1]  #number of examples
    layer_dims = [X.shape[0], 10, 5, 1]
    
    """
    Initialize parameters dictionary
    """
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)
        
    """
    Loop (gradient decent)
    """
    for i in range(0, num_iterations):
        #Forward propagation: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
        a3, cache = forward_propagation(X, parameters)
        
        #compute loss
        cost = compute_loss(a3, Y)
        
        #Backward propagation
        grads = backward_propagation(X, Y, cache)
        
        #Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        #Print loss
        if print_cost and i % 1000 == 0:
            print("Cost after iteration{}: {}".format(i, cost))
            costs.append(cost)
    
    """
    Plot the loss
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    return parameters

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
        layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                        b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters



"""
The following code train the model on 15000 iterations using zeros initialization
"""
parameters = model(train_X, train_Y, initialization = "zeros")
print("On the train set: ")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set: ")
predictions_test = predict(test_X, test_Y, parameters)

#print ("predictions_train = " + str(predictions_train))
#print ("predictions_test = " + str(predictions_test))

"""
#don't know what the problem is here
plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
"""

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.ran









