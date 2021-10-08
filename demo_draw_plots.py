import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d

x = [5,50,100,1000,10000,100000,1000000]
y = [0.00638,0.02254,0.04358,0.2988,4.21784,95.2938,6217.74]
plt.plot(x, y, 'b:')
plt.show()