#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:51:53 2018

@author: dingqingpeng
"""
import matplotlib.pyplot as plt
import numpy as np

a = np.abs(-4)
print('Hello, ' + str(a))

plt.scatter([1, 2, 3, 4], [1, 2, 3, 4], s=40)
plt.show()
plt.scatter([1, 2, 3, 4], [4, 3, 2, 1], s=40)
plt.show()
