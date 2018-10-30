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
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
Load data
"""
train_X, train_Y, test_X, test_Y = load_2D_dataset()


def model(X, Y, learning_rate = 0.3, num_iterations = 3e4, print_cost = True, plot_loss = True, lambd = 0, keep_prob = 1):
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
    
    layers_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layers_dims)
    return parameters
    