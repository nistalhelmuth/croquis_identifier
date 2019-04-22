import numpy as np

def cost_and_gradient():
    return 0

def linear_cost(theta, x, y):
    t0, t1 = theta
    h = t0 + x * t1
    return ((h - y)**2).sum()

