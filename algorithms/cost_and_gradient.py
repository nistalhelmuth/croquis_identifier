import numpy as np
from feed_forward import feed_forward

def calculate_phi(A, y, thetas):
    
    phi3 = (y - A[3])
    phi2 = np.matmul( thetas[2].T, phi3 ) * (A[2]) * (1-A[2]) 
    phi1 = np.matmul( thetas[1].T, phi2[1:] ) * (A[1]) * (1-A[1]) 
    phi0 = np.matmul( thetas[0].T, phi1[1:] ) * (A[0]) * (1-A[0])
    phis = (phi0,phi1,phi2,phi3)

    #update_delta
    delta3 = np.matmul(A[3], phis[2].T)
    delta2 = np.matmul(A[2][1:], phis[1].T)
    delta1 = np.matmul(A[1][1:], phis[0].T)

    return np.array((delta1, delta2, delta3))

# x: imagen para entrenar [tamaño de imagen, cantidad de imagenes]
# thetas: pesos asignados (arreglo de pesos)
# y: lo que debería de ser [diez, cantidad de imagenes]
# return: costo (arreglo de cambios de peso)
def calculate_deltas(x, y, thetas):
    A = feed_forward(x.reshape(-1,1), thetas)
    deltas = calculate_phi(A, y.reshape(-1,1),thetas)
    return deltas

def cost_and_gradient(X, spected_results, thetas):
    np.vectorize(calculate_deltas)
    #new_thetas = calculate_deltas(X,spected_results,thetas).sum(axis=0)

    theta1 = np.zeros((10,785))
    theta2 = np.zeros((10,11))
    theta3 = np.zeros((3,11))
    new_thetas = np.array([theta1, theta2, theta3])
    X_split = np.split(X,3)
    for i in range(0,3):#cuantas categorias
        for o in range(0,20):#imagen por categoria
            new_thetas += np.apply_along_axis(calculate_deltas, 1,X_split[i],spected_results.T[i*2+o],thetas).sum(axis=0)
    
    #new_thetas = np.apply_along_axis(calculate_deltas, 1,X,spected_results[0],thetas).sum(axis=0)
    return new_thetas

