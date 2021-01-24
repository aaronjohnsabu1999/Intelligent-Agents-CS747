import sys, math, random
import numpy as np
import matplotlib.pyplot as plt
from operator import add 
from copy import deepcopy as dcopy


Winds     = [[[0,0],[ 0,0],[0, 0],[-1, 0],[-1,0],[-1,0],[-2,0],[-2,0],[-1,0],[0,0]] for i in range(7)]
NormMoves = [ [0,1],       [-1,0],        [0,-1],       [ 1,0]]
KingMoves = [ [0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1],[ 1,0],[ 1,1]]
Options = [['Sarsa', NormMoves, False, 'Non-stochastic Sarsa with 4 moves'],
           ['Sarsa', KingMoves, False, 'Non-stochastic Sarsa with 8 moves'],
           ['Sarsa', KingMoves, True, 'Stochastic Sarsa with 8 moves'],
           ['Expected Sarsa', NormMoves, False, 'Non-stochastic Expected Sarsa with 4 moves'],
           ['Q-Learning', NormMoves, False, 'Non-stochastic Q-Learning with 4 moves']]
NumEpisodes   =  170
Start         = (3,0)
End           = (3,7)
Epsilon       = 0.1
Discount      = 1
LearningRate = 0.4


def update(function, learningRate, discount, Qt, Qt1, st, at, rt, st1, **kwargs):
  if 'at1' in kwargs:
    at1 = kwargs['at1']
  if 'pit' in kwargs:
    pit = kwargs['pit']
  target = rt
  if function == 'Q-Learning':
    target += discount * np.max(Qt1)
  elif function == 'Sarsa':
    target += discount * Qt1[at1]
  elif function == 'Expected Sarsa':
    target += discount * np.sum([pit[a]*Qt1[a] for a in range(len(Qt1))])
  return (1 - learningRate) * Qt + learningRate * target


def algorithm(moves, function, numEpisodes, start, end, epsilon, discount, learningRate, stochastic = False):
  TIME_STEPS    = [0]
  Rewards       = []
  Q             = [[[-1e10 for action in range(len(moves))] for stateX in range(10)] for stateY in range(7)]
  for stateY in range(7):
    for stateX in range(10):
      for action in range(len(moves)):
        if stateY+moves[action][0] in range(7) and stateX+moves[action][1] in range(10):
          Q[stateY][stateX][action] = 0
  
  time_steps = 0
  
  ## Loop for each episode
  for episode in range(numEpisodes):
    ## Initialize S
    state   = start
    rewards = 0
    
    ## Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    commPol = epsilon/(len(moves)-Q[state[0]][state[1]].count(-1e10))
    policy  = [commPol for i in range(len(moves))]
    policy[np.argmax(Q[state[0]][state[1]])] += 1.0 - epsilon
    for a in range(len(moves)):
      if Q[state[0]][state[1]][a] == -1e10:
        policy[a] = 0
    actionNum  = random.choices(range(len(moves)), weights = policy)[0]
    action     = moves[actionNum]
      
    ## Loop for each step of episode until S is terminal
    while state != end:
      ## EXTRA: Obtain wind values
      wind = dcopy(Winds[state[0]][state[1]])
      if stochastic:
        wind[0] += random.choice([-1,0,1])
      
      ## Take action A, observe R = -1, S'
      new_state      = (max(min(state[0]+action[0]+wind[0], 6), 0), max(min(state[1]+action[1]+wind[1], 9), 0))
      if new_state != end:
        rewards  += -1
      
      ## Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
      new_commPol = epsilon/(len(moves)-Q[new_state[0]][new_state[1]].count(-1e10))
      new_policy  = [new_commPol for i in range(len(moves))]
      new_policy[np.argmax(Q[new_state[0]][new_state[1]])] += 1.0 - epsilon
      for a in range(len(moves)):
        if Q[new_state[0]][new_state[1]][a] == -1e10:
          new_policy[a] = 0
      new_actionNum = random.choices(range(len(moves)), weights = new_policy)[0]
      
      kwargs    = {}
      if function == 'Sarsa':
        kwargs['at1'] = new_actionNum
      elif function == 'Expected Sarsa':
        kwargs['pit'] = new_policy
      
      ## Q(S, A) update
      Q[state[0]][state[1]][actionNum] = round(update(function, learningRate, discount, Q[state[0]][state[1]][actionNum], Q[new_state[0]][new_state[1]], state, action, -1, new_state, **kwargs), 4)
      
      ## S=S'; A = A'
      state       = new_state
      actionNum   = new_actionNum
      action      = moves[actionNum]
      commPol     = new_commPol
      policy      = new_policy
      time_steps += 1
    
    Rewards.append(rewards)
    TIME_STEPS.append(time_steps)
  
  return Q, Rewards, TIME_STEPS


def path(Q, rewards, start, end, moves):
  state   = start
  states  = [state]
  actions = []
  while state != end:
    wind       = Winds[state[0]][state[1]]
    action     = moves[np.argmax(Q[state[0]][state[1]])]
    state      = (max(min(state[0]+action[0]+wind[0], 6), 0), max(min(state[1]+action[1]+wind[1], 9), 0))
    states.append(state)
    actions.append(action)
  return states, actions


def main(options, numEpisodes, start, end, epsilon, discount, learningRate):
  for function, moves, stochastic, label in options:
    tAvgs = []
    QAvgs = []
    rAvgs = []
    for seed in range(10):
      random.seed(seed)
      Q, Rewards, Time_Steps = algorithm(moves, function, numEpisodes, start, end, epsilon, discount, learningRate, stochastic = stochastic)
      if not len(tAvgs):
        tAvgs = np.array(Time_Steps)/10.0
        QAvgs = np.array(Q)/10.0
        rAvgs = np.array(Rewards)/10.0
      else:
        tAvgs = tAvgs + np.array(Time_Steps)/10.0
        QAvgs = QAvgs + np.array(Q)/10.0
        rAvgs = rAvgs + np.array(Rewards)/10.0
        
    if not stochastic:
      print()
      print(label)
      States, Actions = path(QAvgs, rAvgs, start, end, moves)
      print(States)
      print(Actions)
    
    plt.plot(tAvgs, range(len(tAvgs)), label = label)

  plt.legend()
  plt.show()
  plt.savefig("Averaged_Comparison.png")

main(Options, NumEpisodes, Start, End, Epsilon, Discount, LearningRate)