#!/usr/bin/env python3
"""
Optimization of Hyperparameteres 
using  Gaussian Process Optimization
"""
import GPyOpt as GPyOpt
import numpy as np

"""
name=input("Insert the Name of Hyperparameter to optimize: ")
min=float(input("Insert the min limit: "))
min_Y = float(input("Insert the output for min limit: "))
max=float(input("Insert the max limit: "))
max_Y = float(input("Insert the output for max limit: "))
mode=float(input("Insert (1) to Minimize or (2) to Maximize"))
"""
name = 'momentum'
min = 0.01
min_Y = 0.43150
max = 0.99
max_Y = 0.35180
mode = 2

domain =[{'name': name, 'type': 'continuous', 'domain': (min, max)}]
X_step = np.array([min, max])
Y_step = np.array([min_Y, max_Y])
x_next = 0
while not (x_next in X_step):
    bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step)
    x_next = bo_step.suggest_next_locations()
    print("Next value is:", x_next)
    y_next=float(input("Please enter the function value for above value: "))
   
    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

print("You have reached the Max / Min value with X = ", x_next)
print("The values in X were:", X_step)
