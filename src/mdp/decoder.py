import argparse

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid")
    parser.add_argument("--value_policy")
    args = parser.parse_args()

    gridfile              = args.grid
    value_and_policy_file = args.value_policy
    with open(gridfile) as f:
        grid = f.readlines()
    grid = [[int(j) for j in grid[i][:-1].split()] for i in range(len(grid))]
    
    value  = []
    policy = []
    with open(value_and_policy_file) as f:
        out = f.readlines()
        out = [out[i].split() for i in range(len(out))]
        for i in range(len(out)):
            value.append(float(out[i][0]))
            policy.append(int(out[i][1]))
    
    return grid, value, policy

def maze(grid):
    states   = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 1:
                states.append((i,j))
            if grid[i][j] == 2:
                start = len(states)-1
            if grid[i][j] == 3:
                end   = len(states)-1
    return states, start, end

def decode(states, start, end, policy):
    translated = ['N', 'S', 'E', 'W']
    stateNum   = start
    state      = states[stateNum]
    actions    = []
    while stateNum != end:
        actionNum = policy[stateNum]
        action    = translated[actionNum]
        neighbors  = [(state[0]-1,state[1]),(state[0]+1,state[1]),(state[0],state[1]+1),(state[0],state[1]-1)]
        state     = neighbors[actionNum]
        stateNum  = states.index(state)
        actions.append(action)
    return actions

grid, value_function, policy = init()
states, start, end = maze(grid)
path = decode(states, start, end, policy)
for direction in path[:-1]:
    print(direction, end = ' ')
print(direction[-1])