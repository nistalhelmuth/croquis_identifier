import numpy as np
from PIL import Image
from feed_forward import feed_forward
from calculate_phi import calculate_phi

def update_delta(deltas,phis,A):
    for a in A:
        deltas[i] += phis[i+1]*a[i].T
    return deltas

def back_propagation(
    img, 
    spected_result, 
    theta1,
    theta2,
    deltas
    ):
    m = 1
    for i in range(0,m):
        A,H = feed_forward(img, theta1, theta2)
        phis = calculate_phi(A, H,spected_result)
        #D = update_delta(deltas, phis, A)
    return D/m

#solo para circulo
img = Image.open("../images/triangles/1.jpg")
img = np.array(img).reshape(-1,1)
m,n = img.shape
spected_result = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #para la primera salida
theta1 = np.ones(10).reshape(-1,1)
theta2 = np.ones(10).reshape(-1,1)
deltas = np.zeros(10*10).reshape(-1,1)

back_propagation(img,spected_result,theta1,theta2,deltas)



'''
  por cada tipo de imagen 
    por cada imagen del tipo
      feed forward
      back propagation
      update weights
  save delta
'''