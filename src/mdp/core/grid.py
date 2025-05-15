#!/usr/bin/env python3
# grid.py


class Grid:
    def __init__(self, gridfile):
        self.gridfile = gridfile
        self.grid = self.load_grid()

    def load_grid(self):
        try:
            with open(self.gridfile) as f:
                grid = f.readlines()
            return [[int(j) for j in grid[i].strip().split()] for i in range(len(grid))]
        except FileNotFoundError:
            print(f"Error: {self.gridfile} not found.")
            raise
        except ValueError as e:
            print(f"Error reading the grid file {self.gridfile}: {e}")
            raise
