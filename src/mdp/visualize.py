import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class MazeVisualizer:
    def __init__(self, grid_path, path_path=None, output="gridFile.png"):
        self.grid = np.loadtxt(grid_path, dtype=int)
        self.path_path = path_path
        self.output = output
        self.rows, self.cols = self.grid.shape
        self.start_pos = self._find_start()

    def _find_start(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 2:
                    return (i, j)
        return None

    def _apply_path(self):
        x, y = self.start_pos
        self.grid[x][y] = 3

        with open(self.path_path) as f:
            steps = deepcopy(f.readlines()[-1].split())

        for move in steps:
            if move == "N":
                x -= 1
            elif move == "S":
                x += 1
            elif move == "E":
                y += 1
            elif move == "W":
                y -= 1
            self.grid[x][y] = 2

        self.grid[x][y] = 4

    def _convert_solution_markers(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 2:
                    self.grid[i][j] = 3
                elif self.grid[i][j] == 3:
                    self.grid[i][j] = 4

    def render(self):
        if self.path_path:
            self._apply_path()
        else:
            self._convert_solution_markers()

        plt.figure(figsize=(10, 5))
        plt.imshow(self.grid, cmap=plt.cm.CMRmap, interpolation="nearest")
        plt.xticks([]), plt.yticks([])
        plt.savefig(self.output)
        print(f"✅ Maze image saved to: {self.output}")
