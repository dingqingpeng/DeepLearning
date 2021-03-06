#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:54:35 2018

@author: dingqingpeng

Description:
    Second assignment of "Improving Deep Neural Networks"
    Use regularization in the deep learning models.
    
    Problem Statment:
        You have just been hired as an AI expert by the French Football Corporation.
        They would like you to recommend positions where France's goal keeper should kick the ball...
        so that the French team's players can then hit it with their head.
    Goal:
        Use a deep learning model to find the positions on the field where the goalkeeper should kick the ball.
"""

import numpy as np
import matplotlib.pyplot as plt
from reg_utils import initialize_parameters, predict_dec# load_2D_dataset, plot_decision_boundary, sigmoid, relu, 
#from reg_utils import # forward_propagation, compute_cost, backward_propagation, predict, update_parameters
from my_reg_utils import sigmoid, relu, load_2D_dataset
from my_reg_utils import forward_propagation, forward_propagation_with_dropout
from my_reg_utils import backward_propagation, backward_propagation_with_regularization, backward_propagation_with_dropout
from my_reg_utils import update_parameters
from my_reg_utils import predict, plot_decision_boundary
from my_reg_utils import compute_cost, compute_cost_with_regularization
from testCases import *

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset(plot = False)

def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, plot_loss = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
        learning_rate -- learning rate of the optimization
        num_iterations -- number of iterations of the optimization loop
        print_cost -- If True, print the cost every 10000 iterations
        plot_loss -- If True, plot the loss figure
        lambd -- regularization hyperparameter, scalar
        keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
        parameters -- parameters learned by the model. They can then be used to predict.
    """
    
    grads = {}
    costs = []
    layers_dims = [X.shape[0], 20, 3, 1]
    
    #Initialize parameters dictionary
    parameters = initialize_parameters(layers_dims)
    
    #Loop
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
        
        # Backward propagation
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 2000 iterations
        if print_cost and i % 200 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        # Log cost every 1000 iterations
        costs.append(cost)
        
    # Plot the cost
    if plot_loss:
        plt.plot(costs)
        plt.ylabel('costs')
        plt.xlabel('iterations(x 1000)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()
        
    return parameters

# Train model on the train set
parameters = model(train_X, train_Y, print_cost = False, lambd = 0, keep_prob = 0.86)
print("On the training set: ")
predictions_train = predict(train_X, train_Y, parameters, print_results = True)
print("On the test set: ")
# Make predictions on the test set
predictions_test = predict(test_X, test_Y, parameters, print_results = True)

# Plot decision boundary
plt.title("Model with L2 regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
#plot_decision_boundary(train_X, train_Y, parameters)
plot_decision_boundary(test_X, test_Y, parameters)




