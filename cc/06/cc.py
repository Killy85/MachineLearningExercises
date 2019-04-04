import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd 
import time
import datetime

points = [['2019-03-16',81682.0],['2019-03-18',81720.0],['2019-03-20',81760.0],['2019-03-24',81826.0],['2019-03-25',81844.0],['2019-03-26',81864.0],
['2019-03-27',81881.0],['2019-03-28',81900.0],['2019-03-30',81933.0],['2019-04-03',82003.0]]
points = list(map(lambda x : [time.mktime(datetime.datetime.strptime(x[0], "%Y-%m-%d").timetuple()),x[1]], points))
#df = pd.read_csv("exercices/01/Consumo_cerveja.csv") 
#points = [[elem[0], elem[1]] for elem in df.iloc[:,[1,6]].as_matrix()]

X,y= np.array([elem[0] for elem in points]).reshape(-1, 1), np.array([elem[1] for elem in points])

reg = LinearRegression().fit(X, y)



x_print, y_print = [elem[0] for elem in points],[elem[1] for elem in points]

x = np.arange(1552490800,1554442400,20)
y_hat = [reg.predict(np.array([[elem]]).reshape(-1, 1)) for elem in x]

tommorow = time.mktime(datetime.datetime.strptime('2019-04-04', "%Y-%m-%d").timetuple())
tommorow_pred = reg.predict(np.array([tommorow]).reshape(-1, 1))


plt.figure(1)
sub_1 = plt.subplot(211)
plt.plot(x,y_hat, color='blue')
plt.scatter(X, y)
plt.xlabel("Timestamp of the date")
plt.ylabel('Electrical counter value')
plt.scatter(tommorow,tommorow_pred, color='red')


plt.show()


