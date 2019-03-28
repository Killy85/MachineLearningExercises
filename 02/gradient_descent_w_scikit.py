import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd 

points = [[0,4], [1,5], [2,3],[3,5], [4,8], [5,2], [6,1], [7,1], [8,3],[9,1], [10,3]]
df = pd.read_csv("exercices/01/Consumo_cerveja.csv") 
points = [[elem[0], elem[1]] for elem in df.iloc[:,[1,6]].as_matrix()]

X,y= np.array([elem[0] for elem in points]).reshape(-1, 1), np.array([elem[1] for elem in points])

reg = LinearRegression().fit(X, y)



x_print, y_print = [elem[0] for elem in points],[elem[1] for elem in points]

x = np.arange(10,30,0.1)
y_hat = [reg.predict(np.array([[elem]]).reshape(-1, 1)) for elem in x]

print(reg.score(X, y),
        reg.coef_,
        reg.intercept_,
        reg.predict(np.array([[3]]).reshape(-1, 1)))

plt.figure(1)
sub_1 = plt.subplot(211)
plt.scatter(X, y)

plt.plot(x,y_hat)

plt.show()