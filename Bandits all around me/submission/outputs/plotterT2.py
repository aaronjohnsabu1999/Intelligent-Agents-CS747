import sys
import math
import numpy as np
import warnings
from numpy import random
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category = RuntimeWarning)

algos    = ['thompson-sampling', 'thompson-sampling-with-hint']
horizons = ['100', '400', '1600', '6400', '25600', '102400']

def dicter(values, fname):
    with open(fname) as f:
        content  = [line for line in f.read().split("\n")]
    
    for i in range(len(content)):
        line     = content[i].split(", ")
        
        instance = int(line[0][15])
        algo     = str(line[1])
        horizon  = line[4]
        sample   = float(line[5])
        
        try:
            values[algo][instance-1][horizon][0].append(sample)
        except:
            values[algo][instance-1][horizon] = [[sample], sample]
        values[algo][instance-1][horizon][1]  = np.mean(values[algo][instance-1][horizon][0])
    return values

def averager(values, algos, horizons):
    avg = [[[] for j in range(3)] for i in range(len(values))]
    al = 0
    for algo in algos:
        for instance in range(3):
            avg[al][instance] = [values[algo][instance][horizon][1] for horizon in horizons]
        al += 1
    return avg

def plotter(avg):
    plotStyles = ['-or', '-oy', '-ob', '-og']
    for instance in range(3):
        for algo in range(len(avg)):
            plt.plot(horizons, avg[algo][instance], plotStyles[algo], label = algos[algo])
        plt.legend(loc='best')
        plt.show()

values = {}
fname    = 'outputDataT2.txt'
for algo in algos:
    values[algo] = [{}, {}, {}]
values = dicter(values, fname)
plotter(averager(values, algos, horizons))

#for algo in range(4):
    #print(algos[algo])
    #print(avg[algo])