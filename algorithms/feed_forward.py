import numpy as np
import math

def sigmoidfunc(x):
    return 1/(1 + math.exp(-x))

sigmoid = np.vectorize(sigmoidfunc)

def feed_forward(X, theta1, theta2): #tres capas (una oculta)
    A1 = np.vstack((
        1,
        X
    ))
    A2 = np.vstack((
        1,
        sigmoid(np.sum(np.matmul(theta1,A1.T), axis = 1, keepdims = True))
    ))
    A3 = sigmoid(np.sum( np.matmul(theta2,A2.T), axis = 1, keepdims = True))
    return (A1, A2, A3)