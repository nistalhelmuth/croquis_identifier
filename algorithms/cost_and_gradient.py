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

    return (delta1, delta2, delta3)

def cost_and_gradient(thetas):
    theta1 = np.zeros(thetas[0].shape)
    theta2 = np.zeros(thetas[1].shape)
    theta3 = np.zeros(thetas[2].shape)
    gradient = (theta1, theta2, theta3)

    with open('../csvFiles/results.csv') as results_file:
        with open('../csvFiles/trainimages.csv') as images_file:
            results_lines=results_file.readlines()
            image_lines=images_file.readlines()
            for i in range(0,len(image_lines)):
                spected_result = np.fromstring(results_lines[i], dtype=float, sep=',')
                image = np.fromstring(image_lines[i], dtype=float, sep=',')
                A = feed_forward(image.reshape(-1,1), thetas)
                deltas = calculate_phi(A, spected_result.reshape(-1,1),thetas)
                gradient = (gradient[0]+deltas[0],gradient[1]+deltas[1],gradient[2]+deltas[2])
    #new_thetas = np.apply_along_axis(calculate_deltas, 1,X,spected_results[0],thetas).sum(axis=0)
    return gradient

