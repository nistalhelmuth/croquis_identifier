import numpy as np
from cost_and_gradient import cost_and_gradient

norm = lambda v: ((v ** 2 ).sum()) ** 0.5

def gradient_descent(
    X,
    y,
    thetas,
    alpha=0.0001,
    treshold=0.000001,
    max_iter=100000
    ):
    gradient,i = thetas,0
    normalization = norm(np.hstack((gradient[0].ravel(),gradient[1].ravel(),gradient[2].ravel())))
    while i < max_iter and normalization >= treshold:
        print('iteration: ',i)
        gradient = cost_and_gradient(X, y, thetas)
        print(gradient[2])
        thetas += alpha * gradient
        normalization = norm(np.hstack((gradient[0].ravel(),gradient[1].ravel(),gradient[2].ravel())))
        print(normalization)
        i += 1
    return thetas
