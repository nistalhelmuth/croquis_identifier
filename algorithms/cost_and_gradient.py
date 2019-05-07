import numpy as np
from feed_forward import feed_forward

def dRelu(x):
    return 1 * (x>0)

def calculate_phi(A, y, thetas):
    phi4 = (y - A[4])
    #phi3 = np.matmul( thetas[3].T, phi4 ) *  dRelu(A[3])
    phi3 = np.matmul( thetas[3].T, phi4) * (A[3]) * (1-A[3]) 
    phi2 = np.matmul( thetas[2].T, phi3[1:] ) * (A[2]) * (1-A[2]) 
    phi1 = np.matmul( thetas[1].T, phi2[1:] ) * (A[1]) * (1-A[1]) 
    phi0 = np.matmul( thetas[0].T, phi1[1:] ) * (A[0]) * (1-A[0])

    #update_delta
    delta4 = np.matmul(phi4,A[3].T)
    delta3 = np.matmul(phi3[1:],A[2].T)
    delta2 = np.matmul(phi2[1:],A[1].T)
    delta1 = np.matmul(phi1[1:],A[0].T)

    return np.array([delta1, delta2, delta3, delta4])

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
                gradient = gradient + deltas
    return gradient

