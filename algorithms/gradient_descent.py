import numpy as np
import pandas as pd
from cost_and_gradient import cost_and_gradient

norm = lambda v: ((v ** 2 ).sum()) ** 0.5

def gradient_descent(
    thetas,
    alpha=0.01,
    treshold=0.0000001,
    max_iter=1000
    ):
    gradient,i = thetas,0
    norm_value, prev_norm = 99999999, 99999999
    cambios = 0
    while i < max_iter and norm_value >= treshold:
        print('iteration: ',i)
        gradient = cost_and_gradient(thetas)
        norm_value = norm(np.hstack((gradient[0].ravel(),gradient[1].ravel(),gradient[2].ravel(),gradient[3].ravel())))
        print(norm_value)
        if(norm_value < prev_norm):
            prev_norm = norm_value
            prev_gradient = gradient
            prev_thetas = thetas
            thetas += alpha * gradient
        else:
            new_apha = alpha * 0.1
            alpha = new_apha
            print("cambios ",cambios)
            print("cambio de alpha ", alpha)
            pd.DataFrame(prev_thetas[0].reshape(1,-1)).to_csv('../csvFiles/1000/prev_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[1].reshape(1,-1)).to_csv('../csvFiles/1000/prev_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[2].reshape(1,-1)).to_csv('../csvFiles/1000/prev_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[3].reshape(1,-1)).to_csv('../csvFiles/1000/prev_thetas.csv', mode='a',header=False, index=False)
            thetas = prev_thetas + alpha * prev_gradient
        if(i % 100 == 0):
            pd.DataFrame(prev_thetas[0].reshape(1,-1)).to_csv('../csvFiles/1000/_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[1].reshape(1,-1)).to_csv('../csvFiles/1000/_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[2].reshape(1,-1)).to_csv('../csvFiles/1000/_thetas.csv', mode='a',header=False, index=False)
            pd.DataFrame(prev_thetas[3].reshape(1,-1)).to_csv('../csvFiles/1000/_thetas.csv', mode='a',header=False, index=False)
        i += 1
    return thetas
