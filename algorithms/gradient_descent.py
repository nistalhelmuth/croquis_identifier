import numpy as numpy

def gradient_descent(
    X, #(creo) lo que me salio
    y, #(creo) lo que deberia de ser
    theta, #pesos
    cost_and_gradient, #funcion
    alpha=0.01, #cuanto cambio cada peso
    treshold=0.0001, #exactitud
    max_iter=10000, #limite de interaciones
    ):
    last_cost, i = 9999999, 0
    while i < max_iter and norm(cost_and_gradient(X,y,theta)[1]) > treshold:
        cost, gradient = cost_and_gradient(X, y, theta)
        theta = alpha * gradient
        i += 1
        #print('J(i={}):'.format(i),cost)

    return theta