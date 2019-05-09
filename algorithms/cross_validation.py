import numpy as np
import pandas as pd
from feed_forward import feed_forward

ITERATION = 1

with open('../csvFiles/prev_thetas.csv') as thetas_file:
    thetas_lines=thetas_file.readlines()
    theta1 = np.fromstring(thetas_lines[ITERATION*4 + 0], dtype=float, sep=',').reshape(100,785)
    theta2 = np.fromstring(thetas_lines[ITERATION*4 + 1], dtype=float, sep=',').reshape(50,101)
    theta3 = np.fromstring(thetas_lines[ITERATION*4 + 2], dtype=float, sep=',').reshape(10,51)
    theta4 = np.fromstring(thetas_lines[ITERATION*4 + 3], dtype=float, sep=',').reshape(10,11)
    thetas = np.array([theta1, theta2, theta3, theta4])
    

goods = 0
bads = 0
with open('../csvFiles/twickresults.csv') as results_file:
    with open('../csvFiles/twickimages.csv') as images_file:
        results_lines=results_file.readlines()
        image_lines=images_file.readlines()
        for i in range(0,len(image_lines)):
            spected_result = np.fromstring(results_lines[i], dtype=float, sep=',')
            image = np.fromstring(image_lines[i], dtype=float, sep=',')
            A = feed_forward(image.reshape(-1,1), thetas)
            if (np.argmax(A[3]) == np.argmax(spected_result)):
                goods += 1
            else:
                bads += 1

print(goods)
print(bads)
total=bads+goods
print(goods/total * 100)


'''
ITERACION 0:
    sigmoide normalizado:        10% ~235 iteraciones
ITERACION :
    sigmoide normalizado * 3:   >235 iteraciones
ITERACION 1:
    extendiendo una hidden:      10% 315 iteraciones
ITERACION 2:
    agregando una hidden layer:  10% 322 iteraciones
ITERACION 3:
    cambiando sigmoide por Relu:              
ITERACION 4:
    cambiando Relu por softmax:              
ITERACION 5:
'''

