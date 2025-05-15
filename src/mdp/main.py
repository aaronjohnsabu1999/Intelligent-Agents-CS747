#!/usr/bin/env python3
# main.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time

from core.solver import MDPSolver
from core.encoder import Encoder
from core.decoder import Decoder
from core.utils import display_path_on_grid


class MDPPipeline:
    """
    Core MDP solving pipeline that can be reused by different scripts.
    """

    def __init__(self, algorithm, temp_dir="tmp", keep_tmp=False):
        self.algorithm = algorithm
        self.temp_dir = temp_dir
        self.keep_tmp = keep_tmp

        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)

        # Set up file paths
        self.mdp_path = f"{self.temp_dir}/mdp.txt"
        self.value_path = f"{self.temp_dir}/value.txt"
        self.policy_path = f"{self.temp_dir}/policy.txt"

    def solve_grid(self, grid_file, visualize=True):
        """
        Solve a maze grid using the MDP approach.

        Args:
            grid_file: Path to the grid file
            visualize: Whether to display the path visually

        Returns:
            tuple: (V, pi, actions, time_taken)
                V: Value function
                pi: Policy
                actions: List of actions forming the path
                time_taken: Time taken to solve
        """
        start_time = time.time()

        try:
            # Step 1: Encode grid to MDP
            print("📦 Encoding grid to MDP...")
            encoder = Encoder(grid_file)
            mdp_data = encoder.encode()
            with open(self.mdp_path, "w") as f:
                f.write(mdp_data)

            # Step 2: Solve MDP
            print(f"🧠 Solving MDP using {self.algorithm.upper()}...")
            solver = MDPSolver(mdp_input_path=self.mdp_path, algorithm=self.algorithm)
            V, pi = solver.solve()

            # Write value and policy files
            with open(self.value_path, "w") as f:
                out = []
                for v in V:
                    out.append(f"{v}")
                f.writelines([f"{line}\n" for line in out])

            with open(self.policy_path, "w") as f:
                out = []
                for i, p in enumerate(pi):
                    out.append(f"{i} {p}")
                f.writelines([f"{line}\n" for line in out])

            # Step 3: Decode policy to path
            print("🔄 Decoding policy to path...")
            decoder = Decoder(grid_file, self.value_path, self.policy_path)
            actions = decoder.decode()

            # Format path as a string for display
            path = " ".join(actions)
            print("\n📍 PATH:")
            print(path)

            # Step 4: Visualize path on grid if requested
            if visualize:
                grid = np.loadtxt(grid_file, dtype=int)
                display_path_on_grid(grid, path)

            time_taken = time.time() - start_time
            print(f"⏱️ Time taken: {time_taken:.3f} sec")

            return V, pi, actions, time_taken

        finally:
            # Clean up temporary files unless instructed to keep them
            if not self.keep_tmp and os.path.exists(self.temp_dir):
                print("Cleaning up temporary files...")
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)

    def cleanup(self):
        """Remove temporary directory if it exists"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main(grid_file, algorithm, keep_tmp=False):
    # Create and use the pipeline
    pipeline = MDPPipeline(algorithm, keep_tmp=keep_tmp)
    pipeline.solve_grid(grid_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate, solve, and decode an MDP from a maze grid."
    )
    parser.add_argument("--grid", required=True, help="Path to grid .txt file")
    parser.add_argument(
        "--algorithm",
        choices=["vi", "pi", "lp"],
        default="pi",
        help="Algorithm to use: vi (Value Iteration), pi (Policy Iteration), lp (Linear Programming)",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Keep temporary files after execution"
    )
    args = parser.parse_args()

    main(args.grid, args.algorithm, args.keep_tmp)
