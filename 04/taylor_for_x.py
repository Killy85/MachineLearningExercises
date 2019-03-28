import numpy as np
import matplotlib.pyplot as plt
import sys
# This function estimate the value of e pow x
# using the taylor sequence method
def taylor_expo_for_x(x_val):
    k_max = 12
    total = 0
    x = x_val
    plots = {}

    def fact(nb):
        if(nb >= 1):
            return nb * fact(nb-1)
        else: 
            return 1

    for k in range(0,k_max,1):
        total += x ** k / fact(k)
        plots[k] = total

    x = plots.keys()
    y = plots.values()
    plt.plot(x,y)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        x = int(sys.argv[1])
    except:
        x = 1
    taylor_expo_for_x(x)