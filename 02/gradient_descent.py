from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


def main():
    """
    This function is using the gradient descent algorithm to find a model 
    to fit our points
    """
    next_x = 6  # We start the search at x=6
    gamma = 0.01  # Step size multiplier
    precision = 0.00001  # Desired precision of result
    max_iters = 10000  # Maximum number of iterations

    points = [[0,4], [1,5], [2,3],[3,5], [4,8], [5,2], [6,1], [7,1], [8,3],[9,1], [10,3]]
    #df = pd.read_csv("exercices/01/Consumo_cerveja.csv") 
    #points = [[elem[0], elem[1]] for elem in df.iloc[:,[1,6]].as_matrix()]
    teta_0 = next_0 = 1
    teta_1 = next_1 = 1

    # Derivative function
    df_0 = lambda x, y: reduce(lambda x,y : x+y,[(x + y * a) - b for a,b in points])
    df_1 = lambda x, y: reduce(lambda x,y : x+y,[((x + y * a) - b) * a for a,b in points])

    for i in range(max_iters):
        teta_0 = next_0
        teta_1 = next_1
        next_0 = teta_0 - (gamma/len(points)) * df_0(teta_0,teta_1)
        next_1 = teta_1 - (gamma/len(points)) * df_1(teta_0,teta_1)
        step = next_0 - teta_0
        if abs(step) <= precision:
            break

    plot(points, next_0, next_1)

def plot(points, teta_0, teta_1):
    plt.figure(1)
    plt.subplot(211)
    x,y = [elem[0] for elem in points], [elem[1] for elem in points]
    plt.scatter(x, y)
    x = np.arange(0,10,0.1)
    y = [ teta_0 + teta_1 * elem for elem in x]
    plt.plot(x,y)
    plt.subplot(212)
    x,y = [elem[0] for elem in points], [elem[1] - (teta_0 + teta_1 * elem[0]) for elem in points]
    plt.scatter(x,y)
    x = np.arange(0,10,0.1)
    y = [0]*100

    plt.plot(x,y)
    plt.show()


if __name__ == '__main__':
    main()
    