import numpy as np
from feed_forward import feed_forward

def calculate_phi(A, y, thetas):
    phi4 = (y - A[4])
    phi3 = np.matmul( thetas[3].T[1:], phi4) * (A[3][1:]) * (1-A[3][1:])  
    phi2 = np.matmul( thetas[2].T[1:], phi3) * (A[2][1:]) * (1-A[2][1:]) 
    phi1 = np.matmul( thetas[1].T[1:], phi2) * (A[1][1:]) * (1-A[1][1:]) 
    #update_delta
    delta3 = np.matmul(phi4,A[3].T)
    delta2 = np.matmul(phi3,A[2].T)
    delta1 = np.matmul(phi2,A[1].T)
    delta0 = np.matmul(phi1,A[0].T)

    return np.array([delta0, delta1, delta2, delta3])

def cost_and_gradient(thetas):
    theta1 = np.zeros(thetas[0].shape)
    theta2 = np.zeros(thetas[1].shape)
    theta3 = np.zeros(thetas[2].shape)
    theta4 = np.zeros(thetas[3].shape)
    gradient = np.array([theta1, theta2, theta3, theta4])

    with open('../csvFiles/results.csv') as results_file:
        with open('../csvFiles/trainimages.csv') as images_file:
            results_lines=results_file.readlines()
            image_lines=images_file.readlines()
            for i in range(0,len(image_lines)):
                spected_result = np.fromstring(results_lines[i], dtype=float, sep=',')
                image = np.fromstring(image_lines[i], dtype=float, sep=',')
                A = feed_forward(image.reshape(-1,1), thetas)
                deltas = calculate_phi(A, spected_result.reshape(-1,1),thetas)
                gradient += deltas
    return gradient/len(image_lines)

