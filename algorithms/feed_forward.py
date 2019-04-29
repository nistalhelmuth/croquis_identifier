import numpy as np

def sigmoidfunc(x):
    return 1.0/(1 + np.exp(-x))

def tanh(x):
    return 2*sigmoidfunc(2*x) - 1 

def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()
  
def othersigmoid(x):
    x = softmax(x-784)
    return sigmoidfunc(x)
    #return 1/(1 + np.exp(-0.01*x))

def sigmoid_transformation(theta,A):
    z = np.matmul(theta,A)
    delta = z.max() - z.min()
    x = ((z - z.min() - delta*0.5) * 6)/delta
    return sigmoidfunc(x)

def softmax_transformation(theta,A):
    z = np.matmul(theta,A)
    print(z)
    delta = z.max() - z.min()
    x = ((z - z.min() - delta*0.5) * 6)/delta
    print(x.min())
    print(x.max())
    return softmax(x)

def feed_forward_bias(X, thetas, show = False): #tres capas (una oculta)
    A1 = np.vstack((1,X))
    A2 = np.vstack((1,sigmoid_transformation(thetas[0],A1)))
    A3 = np.vstack((1,sigmoid_transformation(thetas[1],A2)))
    #A4 = softmax_transformation(thetas[2],A3)
    #A2 = np.vstack((1,othersigmoid( np.matmul(thetas[0],A1) )))
    #A3 = np.vstack((1,othersigmoid( np.matmul(thetas[1],A2) )))
    A4 = softmax(np.matmul(thetas[2],A3))
    if show:
        print(A4)
    return (A1, A2, A3, A4)