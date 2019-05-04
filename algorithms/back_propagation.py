import os
import sys
import numpy as np
from PIL import Image
from feed_forward import feed_forward
from gradient_descent import gradient_descent

def binary(x):
    if x:
        return 1
    return 0

np.set_printoptions(threshold=sys.maxsize)
tobinary = np.vectorize(binary)
#pesos
theta1 = np.random.rand(10,785)
theta2 = np.random.rand(10,11)
theta3 = np.random.rand(3,11)
thetas = np.array([theta1, theta2, theta3])


#herramientas
results = np.array([0,0,0]).reshape(-1,1) 
categories = os.listdir("../images/")
#categories = ['SadFace','QuestionMark','Mickey','Egg']
categories = ['Square','Triangle','Circle']
'''
print("Square")
img = Image.open("../cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward(img, thetas, show = True)

print("Triangle")
img = Image.open("../triangulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward(img, thetas, show = True)

print("Circle")
img = Image.open("../circulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward(img, thetas, show = True)
'''
images = np.arange(0)

#train part
for category in range(0,len(categories)):
    spected_result = results.copy()
    spected_result[category] += 1
    image_path = "../images/"+categories[category]
    images_dir = os.listdir(image_path)
    #for instance in images_dir:#por cada imagen en una categoria
    print("abriendo:",categories[category])
    for i in range(0,20):
        #content = Image.open(image_path+"/"+instance)
        content = Image.open(image_path+'/'+images_dir[i])
        if (".bmp" in images_dir[i]):
            img = tobinary(np.array(content)[:,:,0].reshape(1,-1))
        else: 
            img = tobinary(np.array(content).reshape(1,-1))
        if(images.shape[0] == 0):
            images = img
            spected_results = spected_result
        else:
            images = np.vstack((images,img))
            spected_results = np.hstack((spected_results,spected_result))

print(images.shape)
print(spected_results.shape)
thetas = gradient_descent(images, spected_results, thetas)

'''
print("DONE!!!")
#after train
print("Tree")
img = Image.open("../arbol.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)

print("SmileyFace")
img = Image.open("../carafeliz.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)

print("Square")
img = Image.open("../cuadrado.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)

print("Triangle")
img = Image.open("../triangulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)

print("Casa")
img = Image.open("../huevo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)

print("Circle")
img = Image.open("../circulo.bmp")
img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
feed_forward_bias(img, thetas,bias, show = True)


def cost_and_gradient(
    thetas,
    deltas,
    spected_result, 
    image_path,
    ):
    deltas = calculate_deltas(X,thetas,deltas,spected_result)
    
    images = os.listdir(image_path)
    for i in images:
        img = Image.open(image_path+"/"+i)
        if (".bmp" in i):
            img = tobinary(np.array(img)[:,:,0].reshape(-1,1))
        else: 
            img = tobinary(np.array(img).reshape(-1,1))
        A = feed_forward(img, thetas)
        deltas = calculate_phi(A, thetas,spected_result,deltas)
    #thetas = (thetas[0] + deltas[0]/len(images), thetas[1]  + deltas[1]/len(images), thetas[2]+ deltas[2]/len(images))
    return deltas

'''