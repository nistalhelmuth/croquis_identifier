import os
import sys
import csv
import numpy as np
from PIL import Image
import pandas as pd

def binary(x):
    if x<125:
        return 1
    return 0

def create_csv(img_list):
    results = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(-1,1) 
    categories = [ 'Tree', 'Egg',  'MickeyMouse', 'SmileyFace', 'SadFace', 'Square', 'Triangle', 'House', 'Circle','QuestionMark']
    images = np.arange(0)
    for category in range(0,len(categories)):
        spected_result = results.copy()
        spected_result[category] += 1
        image_path = "../images/"+categories[category]
        images_dir = os.listdir(image_path)
        print("abriendo:",categories[category])
        for i in img_list:
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
    return (images,spected_results)


tobinary = np.vectorize(binary)

np.random.seed(0)
lista = np.array(np.arange(1000))

train = np.random.choice(lista, 700, replace=False)
test = np.setdiff1d(lista, train)

twick = np.random.choice(test, 200, replace=False)
final_test = np.setdiff1d(test, twick) # no tocar

print("Creando train")
images, spected_results = create_csv(train)
pd.DataFrame(images).to_csv('../csvFiles/trainimages.csv', mode='w',header=False, index=False)
pd.DataFrame(spected_results.T).to_csv('../csvFiles/trainresults.csv', mode='w',header=False, index=False)
print("Creando twick")
images,spected_results  = create_csv(twick)
pd.DataFrame(images).to_csv('../csvFiles/twickimages.csv', mode='w',header=False, index=False)
pd.DataFrame(spected_results.T).to_csv('../csvFiles/twickresults.csv', mode='w',header=False, index=False)
print("Creando test")
images,spected_results  = create_csv(final_test)
pd.DataFrame(images).to_csv('../csvFiles/testimages.csv', mode='w',header=False, index=False)
pd.DataFrame(spected_results.T).to_csv('../csvFiles/finalresults.csv', mode='w',header=False, index=False)


