import sys
import numpy as np
from PIL import Image
from feed_forward import feed_forward_bias

def calculate_phi(A, thetas,spected_result):
    phi3 = (spected_result - A[3]) ** 2 * -1
    phi2 = np.matmul( thetas[2].T, phi3 ) * A[2] * (1-A[2]) 
    phi1 = np.matmul( thetas[1].T, phi2[1:] ) * A[1] * (1-A[1]) 
    phi0 = np.matmul( thetas[0].T, phi1[1:] ) * A[0] * (1-A[0])
    return (phi0, phi1, phi2, phi3)

def update_delta(deltas,phis,A):
    delta1 = deltas[2] + np.matmul(phis[3],A[2].T)
    delta2 = deltas[1] + np.matmul(phis[2][1:],A[1].T)
    delta3 = deltas[0] + np.matmul(phis[1][1:],A[0].T)
    return (delta3, delta2, delta1)

def tobinary(x):
    if x:
        return 1
    return 0

def back_propagation_bias(
    thetas,
    deltas,
    spected_result, 
    image_path
    ):
    images = os.listdir(path)
    for i in images:#por cada imagen en una categoria
        #img = Image.open("../images/huevo/"+str(1+i)+".bmp")
        img = Image.open(path+i)
        img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
        A = feed_forward_bias(img, thetas)
        phis = calculate_phi(A, thetas,spected_result)
        deltas = update_delta(deltas, phis, A)
    return (deltas[0]/len(image), deltas[1]/len(image),deltas[2]/len(image))

np.set_printoptions(threshold=sys.maxsize)

#pesos
theta1 = np.random.rand(10,785)
theta2 = np.random.rand(10,11)
theta3 = np.random.rand(2,11)
thetas = (theta1, theta2, theta3)

#cambios
delta1 = np.zeros(10*785).reshape(10,785)
delta2 = np.zeros(10*11).reshape(10,11)
delta3 = np.zeros(2*11).reshape(2,11)
deltas = (delta1, delta2, delta3)

#herramientas
categories = ["Square","SmileyFace"]
results = np.array([0,0]).reshape(-1,1) 

#before train
print("Square")
img = Image.open("../images/cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
A = feed_forward_bias(img, thetas)
print(A[3])

print("SmileyFace")
img = Image.open("../images/carafeliz.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
A = feed_forward_bias(img, thetas)

print(A[3])

#train part
for category in range(0,len(categories)):
    print("training ",categories[category],"...")
    spected_result = results
    spected_result[category] += 1
    image_path = "../images/"+categories[category]
    deltas = back_propagation_bias(thetas,deltas, spected_result, image_path)
    thetas = (thetas[0] + deltas[0], thetas[1] + deltas[1], thetas[2] + deltas[2])

#after train
print("Square")
img = Image.open("../images/cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
A = feed_forward_bias(img, thetas)
print(A[3])

print("SmileyFace")
img = Image.open("../images/carafeliz.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
A = feed_forward_bias(img, thetas)