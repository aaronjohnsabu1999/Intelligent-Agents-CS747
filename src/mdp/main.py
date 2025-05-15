#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from solver import MDPSolver
from encoder import encode_grid
from decoder import decode_grid


def main(grid_file, algorithm):
    os.makedirs("temp", exist_ok=True)
    mdp_path = "temp/mdp.txt"
    value_policy_path = "temp/value_policy.txt"

    # Step 1: Encode grid to MDP
    print("📦 Encoding grid to MDP...")
    mdp_data = encode_grid(grid_file)
    with open(mdp_path, "w") as f:
        f.write(mdp_data)

    # Step 2: Solve MDP
    print(f"🧠 Solving MDP using {algorithm.upper()}...")
    solver = MDPSolver(mdp_input_path=mdp_path, algorithm=algorithm)
    V, pi = solver.solve()
    with open(value_policy_path, "w") as f:
        for v, p in zip(V, pi):
            f.write(f"{v:.6f}\t{int(p)}\n")

    # Step 3: Decode policy to path
    print("🔄 Decoding policy to path...")
    path = decode_grid(grid_file, value_policy_path)
    print("\n📍 PATH:")
    print(path.strip())

    # Step 4: Visualize path on grid
    grid = np.loadtxt(grid_file, dtype=int)
    display_path_on_grid(grid, path)


def display_path_on_grid(grid, path):
    grid = grid.copy()
    path_steps = path.strip().split()
    x, y = np.where(grid == 2)
    x, y = int(x[0]), int(y[0])

    dirs = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    for step in path_steps:
        dx, dy = dirs.get(step, (0, 0))
        x += dx
        y += dy
        if grid[x][y] != 3:  # Don't overwrite goal
            grid[x][y] = 7   # Mark path

    plt.imshow(grid, cmap="tab10")
    plt.title("Decoded Path on Maze")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, solve, and decode an MDP from a maze grid.")
    parser.add_argument("--grid", required=True, help="Path to grid .txt file")
    parser.add_argument("--algorithm", choices=["vi", "pi", "lp"], default="pi")
    args = parser.parse_args()

    main(args.grid, args.algorithm)
