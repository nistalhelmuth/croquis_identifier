import numpy as np

def sigmoidfunc(x):
    return 1.0/(1 + np.exp(-x))

def newsigmoid(x):
    return 1/(1 + np.exp(-10*x+5))
  
def othersigmoid(x):
    return 1/(1 + np.exp(-0.01*x))

def feed_forward_bias(X, thetas): #tres capas (una oculta)
    A1 = np.vstack((1,X))
    A2 = np.vstack((1,sigmoidfunc(np.matmul(thetas[0],A1))))
    A3 = np.vstack((1,sigmoidfunc(np.matmul(thetas[1],A2))))
    A4 = sigmoidfunc(np.matmul(thetas[2],A3))
    return (A1, A2, A3, A4)