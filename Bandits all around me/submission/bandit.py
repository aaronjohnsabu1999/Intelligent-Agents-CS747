import sys
import math
import numpy as np
import argparse
import warnings
from numpy import random
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category = RuntimeWarning)

algos = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling', 'thompson-sampling-with-hint']

def init():
    # if not len(sys.argv) == 6:
        # print("Incorrect number of inputs")
        # return (), 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--algorithm")
    parser.add_argument("--randomSeed")
    parser.add_argument("--epsilon")
    parser.add_argument("--horizon")
    args = parser.parse_args()

    err  = 0
    Inst = args.instance
    try:
        Rs = int(args.randomSeed)
        Ep = float(args.epsilon)
        Hz = int(args.horizon)
    except:
        print("Conversion issues. Please check your inputs.")
        return (), 1

    An = len(algos)
    for i in range(len(algos)):
        if args.algorithm == algos[i]:
            An = i
            break
    return (Inst, An, Rs, Ep, Hz), 0


arg, err = init()
if err:
    print("Error confronted! Exiting!")
    exit()

def epsGreedy(In, Ep, Rs, Hz):
    random.seed(Rs)
    picks  = {}
    for arm in In:
        picks[arm] = [0, []]

    for t in range(Hz):
        pick = 0
        if random.random_sample() < Ep:
            pick = random.choice(range(len(In)))
        else:
            avg  = [np.mean(picks[arm][1]) for arm in picks]
            pick = avg.index(max(avg))
        
        arm = In[pick]
        picks[arm][0] += 1
        if random.random_sample() < arm:
            picks[arm][1].append(1)
        else:
            picks[arm][1].append(0)
    
    return picks

def ucb(kl, In, Ep, Rs, Hz):
    def log(num):
        try:
            return math.log(num)
        except:
            return 0

    def ucbCalc(kl, p, u, t):
        if not kl:
            return p + math.sqrt(2*log(t)/u)

        c     = 3
        tol   = 1.0e-4

        start = p
        end   = 1.0
        mid   = (start + end) / 2.0
        final = (log(t) + c*log(log(t))) / u

        while abs(start - end) > tol:
            if p*log(p/mid) + (1-p)*log((1-p)/(1-mid)) > final:
                end   = mid
            else:
                start = mid
            mid = (start + end) / 2.0
        return mid

    random.seed(Rs)
    picks  = {}
    for arm in In:
        picks[arm] = [1, [int(random.random_sample() < arm)], 0]

    for t in range(len(picks), Hz):
        # print(str(t+1) + " in " + str(Hz), end = "\r")
        for arm in picks:
            picks[arm][2] = np.mean(picks[arm][1])
        ucb = [ucbCalc(kl, picks[arm][2], picks[arm][0], t) for arm in picks]
        mxP = In[ucb.index(max(ucb))]
        picks[mxP][0] += 1
        if random.random_sample() < mxP:
            picks[mxP][1].append(1)
        else:
            picks[mxP][1].append(0)
    return picks

def thompson(In, Ep, Rs, Hz):
    random.seed(Rs)
    picks  = {}
    for arm in In:
        picks[arm] = [0, [], 0, 0, 0] # [u, ind.pulls, s, f, p]

    for t in range(Hz):
        # print(str(t+1) + " in " + str(Hz), end = "\r")
        betas = [random.beta(picks[arm][2]+1, picks[arm][3]+1) for arm in picks]
        mxP   = In[betas.index(max(betas))]
        picks[mxP][0] += 1
        if random.random_sample() < mxP:
            picks[mxP][1].append(1)
            picks[mxP][2] += 1
        else:
            picks[mxP][1].append(0)
            picks[mxP][3] += 1
        picks[mxP][4] = picks[mxP][2] / (picks[mxP][2] + picks[mxP][3])

    return picks

def thompson_hint(In, Ep, Rs, Hz):
    random.seed(Rs)
    picks  = {}
    hints  = np.sort(In)
    for arm in In:
        picks[arm] = [0, [], 0, 0, 0] # [u, ind.pulls, s, f, p]
    
    for t in range(Hz):
        if t < 0.2*Hz or random.random_sample() < 0.3:
            betas = [random.beta(picks[arm][2]+1, picks[arm][3]+1) for arm in picks]
            arm   = In[betas.index(max(betas))]
        else:
            trueMax = max(hints)
            phi     = 1e+4
            distr   = random.beta(trueMax*phi, (1-trueMax)*phi)
            emps    = [abs(picks[arm][4]-distr) for arm in picks]
            arm     = In[emps.index(min(emps))]

        picks[arm][0] += 1
        if random.random_sample() < arm:
            picks[arm][1].append(1)
            picks[arm][2] += 1
        else:
            picks[arm][1].append(0)
            picks[arm][3] += 1
        picks[arm][4] = picks[arm][2] / (picks[arm][2] + picks[arm][3])
    return picks

def regretCalc(picks):
    optArm = max(float(arm) for arm in picks)
    Hz = sum([picks[arm][0] for arm in picks])
    regret = Hz * optArm
    for arm in picks:
        regret -= np.sum(picks[arm][1])
    return round(regret, 4)

(Inst, An, Rs, Ep, Hz) = arg
regret = 0

with open(Inst) as f:
    content = f.readlines()
In = [float(content[i]) for i in range(len(content))]

if An == 0:
    regret = regretCalc(epsGreedy(In, Ep, Rs, Hz))
elif An == 1:
    regret = regretCalc(ucb(0, In, Ep, Rs, Hz))
elif An == 2:
    regret = regretCalc(ucb(1, In, Ep, Rs, Hz))
elif An == 3:
    regret = regretCalc(thompson(In, Ep, Rs, Hz))
elif An == 4:
    regret = regretCalc(thompson_hint(In, Ep, Rs, Hz))
print(str(Inst) + ", " + str(algos[An]) + ", " + str(Rs) + ", " + str(Ep) + ", " + str(Hz) + ", " + str(regret))