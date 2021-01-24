import sys
import math
import numpy as np
import warnings
from numpy import random
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category = RuntimeWarning)

epsilons = ['0.0001', '0.001', '0.02', '0.05', '0.1', '0.8', '0.9']
horizons = ['102400']

def dicter(values, fname):
    with open(fname) as f:
        content  = [line for line in f.read().split("\n")]
    
    for i in range(len(content)):
        line     = content[i].split(", ")
        
        instance = int(line[0][15])
        algo     = str(line[1])
        epsilon  = str(line[3])
        horizon  = line[4]
        sample   = float(line[5])
        
        try:
            values[epsilon][instance-1][horizon][0].append(sample)
        except:
            values[epsilon][instance-1][horizon] = [[sample], sample]
        values[epsilon][instance-1][horizon][1]  = np.mean(values[epsilon][instance-1][horizon][0])
    return values

def averager(values, epsilons, horizons):
    avg = [[[] for j in range(3)] for i in range(len(values))]
    al = 0
    for eps in epsilons:
        for instance in range(3):
            avg[al][instance] = [values[eps][instance][horizon][1] for horizon in horizons]
        al += 1
    return avg

def plotter(avg):
    plotStyles = ['-xr', '-xy', '-xb', '-xm', '-xg', '-xc', '-xk']
    for instance in range(3):
        for eps in range(len(avg)):
            plt.plot(horizons, avg[eps][instance], plotStyles[eps], label = epsilons[eps])
        plt.legend(loc='best')
        plt.show()

values = {}
fname    = 'outputDataT3.txt'
for eps in epsilons:
    values[eps] = [{}, {}, {}]
avg = averager(dicter(values, fname), epsilons, horizons)
print(avg)
plotter(avg)