#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np

# Add src/ to import path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mdp.solver import MDPSolver
from mdp.encoder import encode_grid
from mdp.decoder import decode_grid


class MazeValidator:
    def __init__(self, algorithm, input_files):
        self.algorithm = algorithm
        self.input_files = input_files

    def run(self):
        for idx, grid_file in enumerate(self.input_files, 1):
            print(f"\n\n🔍 Test instance {idx} " + "-" * 80)
            start_time = time.time()

            mdp_file = "temp/mdp.txt"
            value_policy_file = "temp/value_policy.txt"

            os.makedirs("temp", exist_ok=True)

            print("📦 Encoding grid...")
            mdp_data = encode_grid(grid_file)
            with open(mdp_file, "w") as f:
                f.write(mdp_data)

            print("🧠 Solving MDP...")
            solver = MDPSolver(
                mdp_input_path=mdp_file,
                algorithm=self.algorithm,
                policy_output_path=None,
                value_output_path=None,
            )
            V, pi = solver.solve()
            with open(value_policy_file, "w") as f:
                for v, p in zip(V, pi):
                    f.write(f"{v:.6f}\t{int(p)}\n")

            print("🔄 Decoding policy to path...")
            shortest_path = decode_grid(grid_file, value_policy_file)

            mistake = self.check_traversal(shortest_path, grid_file)
            if not mistake:
                print("✅ All checks passed. Verifying optimality...")
                self.verify_output(shortest_path, grid_file)

            print("⏱️ Time taken: {:.3f} sec".format(time.time() - start_time))

    def check_traversal(self, path, grid_file):
        grid = np.loadtxt(grid_file, dtype=int)
        steps = path.strip().split()

        x, y = np.where(grid == 2)
        x, y = int(x[0]), int(y[0])
        dirs = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}

        for move in steps:
            if move not in dirs:
                print(f"❌ Invalid move: {move}")
                return True
            dx, dy = dirs[move]
            x += dx
            y += dy
            if grid[x][y] == 1:
                print(f"❌ Hit a wall at ({x}, {y})")
                return True

        gx, gy = np.where(grid == 3)
        return not any(x == ex and y == ey for ex, ey in zip(gx, gy))

    def verify_output(self, path, grid_file):
        sol_file = grid_file.replace("grid", "solution")
        if not os.path.exists(sol_file):
            print("⚠️ No solution file found to compare.")
            return

        with open(sol_file) as f:
            base = f.read().split()
        test = path.strip().split()

        if len(test) > len(base):
            print("⚠️ Your path is longer than optimal.")
        elif len(test) == len(base):
            print("✅ Your path is optimal.")
        else:
            print("❗ Your path is *shorter* than the known optimal?!")
            print("Reference:", base)
            print("Yours:    ", test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze MDP Validator")
    parser.add_argument("--algorithm", choices=["vi", "pi", "lp"], default="pi")
    parser.add_argument(
        "--mazes",
        nargs="+",
        default=[f"data/maze/grid{i}.txt" for i in range(10, 110, 10)],
        help="List of maze grid files",
    )
    args = parser.parse_args()

    validator = MazeValidator(args.algorithm, args.mazes)
    validator.run()
