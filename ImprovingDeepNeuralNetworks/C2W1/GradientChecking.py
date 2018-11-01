# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:44 2018

@author: Andy

Description:
    Third assignment of "Improving Deep Neural Networks"
    Gradient checking.
"""

import numpy as np

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
