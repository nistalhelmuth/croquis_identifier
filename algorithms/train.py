import numpy as np
import pandas as pd
from gradient_descent import gradient_descent


theta1 = np.random.rand(100,785)
theta2 = np.random.rand(50,101)
theta3 = np.random.rand(10,51)
theta4 = np.random.rand(10,11)
thetas = (theta1, theta2, theta3, theta4)
thetas = gradient_descent(thetas)
pd.DataFrame(thetas[0].reshape(1,-1)).to_csv('../csvFiles/thetas.csv', mode='a',header=False, index=False)
pd.DataFrame(thetas[1].reshape(1,-1)).to_csv('../csvFiles/thetas.csv', mode='a',header=False, index=False)
pd.DataFrame(thetas[2].reshape(1,-1)).to_csv('../csvFiles/thetas.csv', mode='a',header=False, index=False)
pd.DataFrame(thetas[3].reshape(1,-1)).to_csv('../csvFiles/thetas.csv', mode='a',header=False, index=False)

#las primeras 2 iteraciones son de 3 thetas
