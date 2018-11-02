#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:44 2018

@author: Andy

Description:
    Third assignment of "Improving Deep Neural Networks"
    Gradient checking.
"""

import numpy as np
from my_reg_utils import sigmoid, relu
from gc_utils import dictionary_to_vector, gradients_to_vector, vector_to_dictionary
from testCases import gradient_check_n_test_case

def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) (J(theta) = theta * x)
    
    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
    
    Returns:
        J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    
    J = x * theta
    
    return J

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta.
    
    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
    
    Returns:
        dtheta -- the gradient of the cost with respect to theta
    """

    dtheta = x
    
    return dtheta

def gradient_check(x, theta, epsilon = 1e-7):
    """
    Check if gradapprox is close enough to the output of backward_propagation()
    
    Arguments:
        x -- a real-valued input
        theta -- our parameter, a real number as well
        epsilon -- tiny shift to the input to compute approximated gradient
        
    Returns:
        difference -- difference between the approximated gradient and the backward propagation gradient
    """
    
    # Compute gradapprox
    J_plus = forward_propagation(x, theta + epsilon)
    J_minus = forward_propagation(x, theta - epsilon)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    # Compute grad
    grad = backward_propagation(x, theta)
    
    # Compute difference
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    # Check if the difference is small enough
    if difference < 1e-7:
        print("The gradient check passed")
    else:
        print("The gradient check failed")
        
    return difference

def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost).
    
    Arguments:
        X -- training set for m examples
        Y -- labels for m examples 
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                        W1 -- weight matrix of shape (5, 4)
                        b1 -- bias vector of shape (5, 1)
                        W2 -- weight matrix of shape (3, 5)
                        b2 -- bias vector of shape (3, 1)
                        W3 -- weight matrix of shape (1, 3)
                        b3 -- bias vector of shape (1, 1)
    
    Returns:
        cost -- the cost function (logistic cost for one example)
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
    
    # Compute cost
    m = X.shape[1]
    logprobs = np.multiply(Y, np.log(A3)) + np.multiply(1 - Y, np.log(1 - A3))
    cost = -1./m * np.nansum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache

def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented.
    
    Arguments:
        X -- input datapoint, of shape (input size, 1)
        Y -- true "label"
        cache -- cache output from forward_propagation_n()
    
    Returns:
        gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
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

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
        parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
        grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
        x -- input datapoint, of shape (input size, 1)
        y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        """
        Attention:
            Use
                theta_plus = np.copy(parameters_values)
            instead of
                theta_plus = parameters_values
            because the second way of assignment actually points two variables ...
            to the same memory segment
        """
        theta_plus = np.copy(parameters_values)
        theta_plus[i] += epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))
        
        theta_minus = np.copy(parameters_values)
        theta_minus[i] -= epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))
        
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    
    # Compute difference
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference > 2e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)