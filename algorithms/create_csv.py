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

def create_csv(img_list,path_image,path_results):
    results = np.array([0,0,0,0,0,0,0,0,0,0]).reshape(-1,1) 
    categories = [ 'Tree', 'Egg',  'MickeyMouse', 'SmileyFace', 'SadFace', 'Square', 'Triangle', 'House', 'Circle','QuestionMark']
    #categories = [ 'QuestionMark']
    for category in range(0,len(categories)):
        spected_result = results.copy()
        spected_result[category] += 1
        image_path = "../images/"+categories[category]
        images_dir = os.listdir(image_path)
        print("abriendo:",categories[category])
        images = np.arange(0)
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
        pd.DataFrame(images).to_csv(path_image, mode='a',header=False, index=False)
        pd.DataFrame(spected_results.T).to_csv(path_results, mode='a',header=False, index=False)
    return (images,spected_results)


tobinary = np.vectorize(binary)

np.random.seed(0) para 1000
lista = np.array(np.arange(1000))
#np.random.seed(1)
#lista = np.array(np.arange(2000))

train = np.random.choice(lista, 800, replace=False)
test = np.setdiff1d(lista, train)

twick = np.random.choice(test, 100, replace=False)
final_test = np.setdiff1d(test, twick) # no tocar

print("Creando train")
path_image='../csvFiles/1000/trainimages.csv'
path_results='../csvFiles/1000/trainresults.csv'
create_csv(train,path_image,path_results)

print("Creando twick")
path_image='../csvFiles/1000/twickimages.csv'
path_results='../csvFiles/1000/twickresults.csv'
create_csv(twick,path_image,path_results)
#
print("Creando test")
path_image='../csvFiles/1000/testimages.csv'
path_results='../csvFiles/1000/testresults.csv'
create_csv(final_test,path_image,path_results)


