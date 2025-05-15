#!/usr/bin/env python3
# encoder.py
from core.grid import Grid


class Encoder:
    def __init__(self, gridfile):
        self.gridfile = gridfile
        self.grid = Grid(gridfile)
        self.maze = self.encode(self.grid)

    def manhattan(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def encode(self, grid=None):
        states = []
        ends = []
        start = None
        end = None
        transitions = []
        state_rewards = -1
        invalid_reward = -1000000
        end_reward = 1000000
        discount = 0.9
        mode = "episodic"
        if grid is None:
            grid = self.grid
        else:
            self.grid = grid

        for i in range(len(grid.grid)):
            for j in range(len(grid.grid[i])):
                if grid.grid[i][j] != 1:
                    states.append((i, j))
                if grid.grid[i][j] == 2:
                    start = len(states) - 1
                if grid.grid[i][j] == 3:
                    end = len(states) - 1
                    ends.append(end)

        # Choose the closest end point
        closest_end = ends[0]
        for check in ends[1:]:
            if self.manhattan(states[check], states[start]) < self.manhattan(
                states[closest_end], states[start]
            ):
                closest_end = check
        end = closest_end

        # Generate transitions
        end_state = states[end]
        for i in range(len(states)):
            state = states[i]
            neighbors = [
                (state[0] - 1, state[1]),
                (state[0] + 1, state[1]),
                (state[0], state[1] + 1),
                (state[0], state[1] - 1),
            ]
            if i == end:
                continue
            for action in range(4):
                neighbor = neighbors[action]
                if neighbor == end_state:
                    transitions.append((i, action, end, end_reward, 1.0))
                elif neighbor in states:
                    transitions.append(
                        (i, action, states.index(neighbor), state_rewards, 1.0)
                    )
                else:
                    transitions.append((i, action, i, invalid_reward, 1.0))

        num_states = len(states)
        num_actions = 4  # 4 possible actions (up, down, left, right)

        # Create a string to represent the MDP data
        mdp_data = f"numStates {num_states}\nnumActions {num_actions}\nstart {start}\nend {end}\n"
        for transition in transitions:
            mdp_data += f"transition {transition[0]} {transition[1]} {transition[2]} {transition[3]} {transition[4]}\n"
        mdp_data += f"mdptype {mode}\ndiscount {discount}\n"

        return mdp_data  # Return MDP data as a string
