#!/usr/bin/env python3
# verify.py
import os
import time
import argparse
import numpy as np

# Import the MDPPipeline class from main.py
from main import MDPPipeline


class MazeValidator:
    """
    Validates MDP solutions for multiple maze grids by comparing with known solutions.
    """

    def __init__(self, algorithm, input_files, keep_tmp=False):
        self.algorithm = algorithm
        self.input_files = input_files
        self.keep_tmp = keep_tmp

        # Create an instance of the MDPPipeline
        self.pipeline = MDPPipeline(algorithm, keep_tmp=keep_tmp)

    def run(self):
        """Run validation on all input files"""
        try:
            for idx, grid_file in enumerate(self.input_files, 1):
                print(f"\n\n🔍 Test instance {idx}: {grid_file}" + "\n" + "-" * 80)

                # Use the pipeline to solve the grid
                _, _, actions, _ = self.pipeline.solve_grid(grid_file, visualize=True)
                path = " ".join(actions)

                # Verify the solution
                mistake = self.check_traversal(path, grid_file)
                if not mistake:
                    print("✅ Path is valid. Verifying optimality...")
                    self.verify_output(path, grid_file)
        finally:
            # Clean up at the end
            self.pipeline.cleanup()

    def check_traversal(self, path, grid_file):
        """Check if the path successfully traverses from start to goal without hitting walls"""
        grid = np.loadtxt(grid_file, dtype=int)
        steps = path.strip().split()

        # Find start position (marked as 2 in the grid)
        start_positions = np.where(grid == 2)
        if len(start_positions[0]) == 0:
            print("❌ No start position (2) found in grid")
            return True

        x, y = int(start_positions[0][0]), int(start_positions[1][0])

        # Direction mappings
        dirs = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

        # Follow the path
        for move in steps:
            if move not in dirs:
                print(f"❌ Invalid move: {move}")
                return True

            dx, dy = dirs[move]
            x += dx
            y += dy

            # Check bounds
            if x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1]:
                print(f"❌ Out of bounds at ({x}, {y})")
                return True

            # Check for walls
            if grid[x][y] == 1:
                print(f"❌ Hit a wall at ({x}, {y})")
                return True

        # Check if we reached the goal (marked as 3 in the grid)
        goal_positions = np.where(grid == 3)
        if len(goal_positions[0]) == 0:
            print("⚠️ No goal position (3) found in grid")
            return True

        reached_goal = False
        for gx, gy in zip(goal_positions[0], goal_positions[1]):
            if x == gx and y == gy:
                reached_goal = True
                break

        if not reached_goal:
            print(f"❌ Did not reach goal. Ended at ({x}, {y})")
            return True

        return False

    def verify_output(self, path, grid_file):
        """Verify that the path is optimal by comparing with solution file"""
        sol_file = grid_file.replace("grid", "solution")
        if not os.path.exists(sol_file):
            print("⚠️ No solution file found to compare.")
            return

        try:
            with open(sol_file) as f:
                base = f.read().strip().split()
            test = path.strip().split()

            if len(test) > len(base):
                print(
                    f"⚠️ Your path is longer than optimal: {len(test)} steps vs {len(base)} steps"
                )
            elif len(test) == len(base):
                print("✅ Your path is optimal.")
            else:
                print(
                    f"❗ Your path is shorter than the known optimal: {len(test)} steps vs {len(base)} steps"
                )

            print(f"Reference path: {' '.join(base)}")
            print(f"Your path:      {' '.join(test)}")

        except Exception as e:
            print(f"⚠️ Error comparing with solution file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze MDP Validator")
    parser.add_argument(
        "--algorithm",
        choices=["vi", "pi", "lp"],
        default="pi",
        help="Algorithm to use: vi (Value Iteration), pi (Policy Iteration), lp (Linear Programming)",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        default=[f"data/maze/grid{i}.txt" for i in range(10, 110, 10)],
        help="List of maze grid files to validate",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Keep temporary files after execution"
    )
    args = parser.parse_args()

    validator = MazeValidator(args.algorithm, args.grid, args.keep_tmp)
    validator.run()
