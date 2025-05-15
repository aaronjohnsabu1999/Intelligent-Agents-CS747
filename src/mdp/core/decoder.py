#!/usr/bin/env python3
# decoder.py
from core.grid import Grid


class Decoder:
    def __init__(self, gridfile, value_path, policy_path):
        self.gridfile = gridfile
        self.grid = Grid(gridfile)
        self.value_path = value_path
        self.policy_path = policy_path
        self.value, self.policy = self.load_value_and_policy()
        self.states, self.start, self.end = self.build_maze()

    def load_value_and_policy(self):
        value, policy = [], []
        with open(self.value_path) as f:
            out = f.readlines()
            out = [out[i].split() for i in range(len(out))]
            for i in range(len(out)):
                value.append(float(out[i][0]))
        with open(self.policy_path) as f:
            out = f.readlines()
            out = [out[i].split() for i in range(len(out))]
            for i in range(len(out)):
                policy.append(int(out[i][1]))
        return value, policy

    def manhattan(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def build_maze(self):
        states = []
        start = end = None
        for i in range(len(self.grid.grid)):
            for j in range(len(self.grid.grid[i])):
                if self.grid.grid[i][j] != 1:
                    states.append((i, j))
                if self.grid.grid[i][j] == 2:
                    start = len(states) - 1
                if self.grid.grid[i][j] == 3:
                    end = len(states) - 1
        return states, start, end

    def get_maze_data(self):
        translated = ["N", "S", "E", "W"]
        states = self.states
        transitions = []
        endstate = states[self.end]
        endReward = 1000000
        stateReward = -1
        invalidReward = -1000000

        for i in range(len(states)):
            state = states[i]
            neighbors = [
                (state[0] - 1, state[1]),
                (state[0] + 1, state[1]),
                (state[0], state[1] + 1),
                (state[0], state[1] - 1),
            ]
            if i == self.end:
                continue
            for action in range(4):
                neighbor = neighbors[action]
                if neighbor == endstate:
                    transitions.append((i, action, self.end, endReward, 1.0))
                elif neighbor in states:
                    transitions.append(
                        (i, action, states.index(neighbor), stateReward, 1.0)
                    )
                else:
                    transitions.append((i, action, i, invalidReward, 1.0))

        mode = "episodic"
        discount = 0.9
        numStates = len(states)
        numActions = 4

        return numStates, numActions, self.start, self.end, transitions, mode, discount

    def decode(self):
        translated = ["N", "S", "E", "W"]
        state_num = self.start
        state = self.states[state_num]
        actions = []

        state_to_idx = {state: idx for idx, state in enumerate(self.states)}

        while state_num != self.end:
            action_num = self.policy[state_num]
            action = translated[action_num]
            actions.append(action)

            neighbors = [
                (state[0] - 1, state[1]),  # N
                (state[0] + 1, state[1]),  # S
                (state[0], state[1] + 1),  # E
                (state[0], state[1] - 1),  # W
            ]
            state = neighbors[action_num]
            try:
                state_num = state_to_idx[state]
            except KeyError:
                raise ValueError(f"Invalid state transition to {state}")

        return actions
