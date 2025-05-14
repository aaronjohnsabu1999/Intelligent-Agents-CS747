import pulp, argparse, time
import numpy as np
from copy import deepcopy

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp")
    parser.add_argument("--algorithm")
    args = parser.parse_args()

    name = args.mdp
    algo = args.algorithm
    with open(name) as f:
        content = f.readlines()
    name = [content[i][:-1].split() for i in range(len(content))]

    mdp  = {}
    for value in name:
        if value[0] == 'transition':
            if 'transition' not in mdp:
                mdp['transition'] = [[[] for a in range(mdp['numActions'])] for s in range(mdp['numStates'])]
            i, j = int(value[1]), int(value[2])
            mdp['transition'][i][j].append((int(value[3]), float(value[4]), float(value[5])))
        elif value[0] == 'episodic':
            mdp['type']   = value[0]
        elif value[0] == 'mdptype':
            mdp['type']   = value[1]
        elif value[0] == 'end':
            mdp['end'] = [int(i) for i in value[1:]]
        elif value[0] == 'discount':
            mdp['discount'] = float(value[1])
        else:
            mdp[value[0]] = int(value[1])
    return mdp, algo

def valueEval(policy, numStates, numActions, tolerance, transition, discount):
    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)
    while(np.linalg.norm(V0-V1) > tolerance):
        V0 = deepcopy(V1)
        for state in range(numStates):
            option    = transition[state][int(policy[state])]
            V1[state] = 0
            for prob in option:
                V1[state] += prob[2]*(prob[1] + discount*V0[prob[0]])
    return [round(state,6) for state in V1]

def actVlEval(V, state, action, transition, discount):
    Q = 0
    option = transition[state][action]
    for s in option:
        Q += s[2]*(s[1] + discount*V[s[0]])
    return round(Q,6)

def valueIter(numStates, numActions, transition, discount, tolerance):    
    V0 = np.ones(numStates)
    V1 = np.zeros(numStates)
    pi = np.zeros(numStates)
    t  = 0
    
    while(np.linalg.norm(V0-V1) > tolerance):
        V0 = deepcopy(V1)
        for state in range(numStates):
            action = 0
            values    = []
            for action in range(numActions):
                try:
                    option = transition[state][action]
                except:
                    continue
                value  = 0
                for prob in option:
                    value += prob[2]*(prob[1] + discount*V0[prob[0]])
                values.append(value)
            V1[state]  = np.max(values)
            pi[state]  = np.argmax(values)
        t = t + 1
    return V1, pi

def linearProg(numStates, numActions, transition, discount, tolerance):
    prob = pulp.LpProblem("ValueFn", pulp.LpMinimize)
    
    decision_variables = []
    for state in range(numStates):
        variable = str('V' + str(state))
        variable = pulp.LpVariable(str(variable))
        decision_variables.append(variable)
    
    total_cost = 0
    for state in range(numStates):
        total_cost += decision_variables[state]
    prob += total_cost
    
    for state in range(numStates):
        for action in range(numActions):
            prob += ( decision_variables[state] >= pulp.lpSum([s[2]*(s[1] + discount*decision_variables[s[0]]) for s in transition[state][action]]))
    prob.writeLP("LinearProgramming.lp")
    
    optimization_result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    V0 = [0 for i in range(numStates)]
    for v in prob.variables():
        index = int(v.name[1:])
        V0[index] = round(v.varValue, 6)

    pi = [0 for i in range(numStates)]
    for state in range(numStates):
        value  = V0[state]
        actVal = [actVlEval(V0, state, action, transition, discount) for action in range(numActions)]
        action = (np.abs(np.asarray(actVal) - value)).argmin() 
        pi[state] = action
    V0 = valueEval(pi, numStates, numActions, tolerance, transition, discount)
    return V0, pi

def policyIter(numStates, numActions, transition, discount, tolerance):    
    pi   = np.array([0 for i in range(numStates)])
    flag = True
    while flag:
        flag = False
        V  = valueEval(pi, numStates, numActions, tolerance, transition, discount)
        for state in range(numStates):
            stateVal = V[state]
            actionIA = -1
            actValIA = stateVal
            for action in range(numActions):
                actVal = actVlEval(V, state, action, transition, discount)
                if abs(actVal - stateVal) < 3e-6:
                    continue
                if actVal > actValIA:
                    actionIA = action
                    actValIA = actVal
            if actionIA != -1:
                flag = True
                pi[state] = actionIA
    return V, pi

mdp, algo = init()

numStates  = mdp['numStates']
numActions = mdp['numActions']
transition = mdp['transition']
discount   = mdp['discount']
tolerance  = 1e-12

if (algo == 'hpi'):
    V0, pi = policyIter(numStates, numActions, transition, discount, 1e-9)
elif (algo == 'lp'):
    V0, pi = linearProg(numStates, numActions, transition, discount, tolerance)
elif (algo == 'vi'):
    V0, pi = valueIter(numStates, numActions, transition, discount, tolerance)

for i in range(len(V0)):
    print('{:.6f}'.format(round(V0[i], 6)) + "\t" + str(int(pi[i])))