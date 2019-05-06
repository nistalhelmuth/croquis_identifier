import numpy as np

def sigmoidfunc(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_transformation(theta,A):
    z = np.matmul(theta,A)
    return sigmoidfunc((z - z.std())/z.mean())

def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()

def softmax_transformation(theta,A):
    z = np.matmul(theta,A)
    return softmax((z - z.std())/z.mean())

def ReLU(x):
    return x * (x > 0)

def feed_forward(X, thetas,show=False): #tres capas (una oculta)
    A1 = np.vstack((1,X))
    A2 = np.vstack((1,sigmoid_transformation(thetas[0],A1)))
    A3 = np.vstack((1,sigmoid_transformation(thetas[1],A2)))
    A4 = np.vstack((1,sigmoid_transformation(thetas[2],A3)))
    #A5 = sigmoid_transformation(thetas[3],A4)
    #A5 = ReLU(np.matmul(thetas[3],A4))
    A5 = softmax(np.matmul(thetas[3],A4))
    return (A1, A2, A3, A4, A5)