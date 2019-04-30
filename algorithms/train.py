import sys
import os
import numpy as np
from PIL import Image
from feed_forward import feed_forward_bias

def calculate_phi(A, thetas,spected_result):
    phi3 = (spected_result - A[3])
    phi2 = np.matmul( thetas[2].T[1:], phi3 ) * (A[2][1:]) * (1-A[2][1:]) 
    #phi1 = np.matmul( thetas[1].T[1:], phi2 ) * (A[1][1:]) * (1-A[1][1:]) 
    #phi0 = np.matmul( thetas[0].T[1:], phi1 ) * (A[0][1:]) * (1-A[0][1:])
    phi1 = np.matmul( thetas[1].T[1:], phi2 ) * (1/np.cosh(A[1][1:])**2)
    phi0 = np.matmul( thetas[0].T[1:], phi1 ) * (1/np.cosh(A[0][1:])**2)
    return (phi0, phi1, phi2, phi3)

def update_delta(deltas,phis,A):
    #delta1 = deltas[2] + np.matmul(A[3],phis[2].T)
    #delta2 = deltas[1] + np.matmul(A[2][1:],phis[1].T)
    #delta3 = deltas[0] + np.matmul(A[1][1:],phis[0].T)
    
    mul1 = np.matmul(phis[3],A[2][1:].T)
    delta3 = deltas[2] + np.hstack((mul1.mean(axis = 1).reshape(-1,1),mul1))
    mul2 = np.matmul(phis[2],A[1][1:].T)
    delta2 = deltas[1] +  np.hstack((mul2.mean(axis = 1).reshape(-1,1),mul2))
    mul3 = np.matmul(phis[1],A[0][1:].T)
    delta1 =  deltas[0] + np.hstack((mul3.mean(axis = 1).reshape(-1,1),mul3))
    return (delta1, delta2, delta3)

def binary(x):
    #return x
    if x:
        return 1
    return 0

def back_propagation_bias(
    thetas,
    deltas,
    spected_result, 
    image_path,
    ):
    images = os.listdir(image_path)
    print("with ",len(images)," samples")
    count = 1
    for i in images:#por cada imagen en una categoria
        img = Image.open(image_path+"/"+i)
        #img = Image.open(image_path+"/1.jpg")
        if (".bmp" in i):
            img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
        else: 
            img = tobinary(np.array(img).reshape(-1,1))
        A = feed_forward_bias(img, thetas)
        phis = calculate_phi(A, thetas,spected_result)
        deltas = update_delta(deltas, phis, A) 
        #thetas = (thetas[0] + deltas[0]/count, thetas[1]  + deltas[1]/count, thetas[2]  + deltas[2]/count)
        count +=1
    #return thetas
    return (thetas[0] - deltas[0]*0.01/len(images), thetas[1]  - deltas[1]*0.01/len(images), thetas[2]- deltas[2]*0.01/len(images))

np.set_printoptions(threshold=sys.maxsize)
tobinary = np.vectorize(binary)
#pesos
theta1 = np.random.randn(10,785)
theta2 = np.random.randn(10,11)
theta3 = np.random.randn(6,11)
#theta1 = np.random.randint(10, size=(10,785))
#theta2 = np.random.randint(10, size=(10,11))
#theta3 = np.random.randint(10, size=(6,11))
thetas = (theta1, theta2, theta3)

#cambios
delta1 = np.zeros(10*785).reshape(10,785)
delta2 = np.zeros(10*11).reshape(10,11)
delta3 = np.zeros(6*11).reshape(6,11)
deltas = (delta1, delta2, delta3)


#herramientas
results = np.array([0,0,0,0,0,0]).reshape(-1,1) 
categories = os.listdir("../images/")
#categories = ["Tree", "SmileyFace","Square","Triangle"]
print(categories)


print("Square")
img = Image.open("../cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

#train part
for category in range(0,len(categories)):
    
    print("training ",categories[category],"...")
    spected_result = results.copy()
    spected_result[category] += 1
    
    image_path = "../images/"+categories[category]
    thetas = back_propagation_bias(thetas,deltas, spected_result, image_path)    

print("DONE!!!")
#after train


print("Tree")
img = Image.open("../arbol.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

print("SmileyFace")
img = Image.open("../carafeliz.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

print("Square")
img = Image.open("../cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

print("Triangle")
img = Image.open("../triangulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

print("Casa")
img = Image.open("../huevo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)

print("Circle")
img = Image.open("../circulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas, show = True)