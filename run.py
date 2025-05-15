import argparse
import subprocess
import sys


def run_bandits(args):
    subprocess.run(["python", "src/bandits/main.py"] + args)


def run_mdp_generate(args):
    subprocess.run(["python", "src/mdp/generate.py"] + args)


def run_mdp_solve(args):
    subprocess.run(["python", "src/mdp/main.py"] + args)

def run_mdp_verify(args):
    subprocess.run(["python", "src/mdp/verify.py"] + args)

def run_mdp_visualize(args):
    subprocess.run(["python", "src/mdp/visualize.py"] + args)

def run_windy_gridworld(args):
    subprocess.run(["python", "src/windy_gridworld/main.py"] + args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Agents Project Runner")
    subparsers = parser.add_subparsers(dest="command")

    # Bandits
    bandit_parser = subparsers.add_parser("bandits", help="Run Bandits experiments")
    bandit_parser.add_argument("--instance", required=True)
    bandit_parser.add_argument("--algorithm", required=True)
    bandit_parser.add_argument("--horizon", required=True)
    bandit_parser.add_argument("--randomSeed", default="0")
    bandit_parser.set_defaults(func=lambda args: run_bandits(sys.argv[2:]))

    # Windy Gridworld
    gridworld_parser = subparsers.add_parser(
        "windy", help="Run Windy Gridworld experiments"
    )
    gridworld_parser.add_argument("--episodes", type=int, default=170)
    gridworld_parser.set_defaults(func=lambda args: run_windy_gridworld(sys.argv[2:]))

    # Generate MDP
    gen_parser = subparsers.add_parser(
        "generate_mdp", help="Generate a random MDP instance"
    )
    gen_parser.add_argument("--num_states", type=int, default=10)
    gen_parser.add_argument("--num_actions", type=int, default=5)
    gen_parser.add_argument("--gamma", type=float, default=0.95)
    gen_parser.add_argument(
        "--mdptype", choices=["continuing", "episodic"], default="episodic"
    )
    gen_parser.add_argument("--rseed", type=int, default=42)
    gen_parser.add_argument("--output_file", required=True)
    gen_parser.set_defaults(func=lambda args: run_mdp_generate(sys.argv[2:]))

    # Solve MDP Maze
    solve_parser = subparsers.add_parser(
        "solve_mdp", help="Solve a maze-based MDP using vi, lp, or pi"
    )
    solve_parser.add_argument("--grid", required=True)
    solve_parser.add_argument("--algorithm", choices=["vi", "pi", "lp"], default="pi")
    solve_parser.add_argument("--keep-tmp", action="store_true")
    solve_parser.set_defaults(func=lambda args: run_mdp_solve(sys.argv[2:]))

    # Verify MDP solutions against known paths
    verify_parser = subparsers.add_parser(
        "verify_mdp", help="Validate maze solution paths using the decoder"
    )
    verify_parser.add_argument(
        "--algorithm",
        choices=["vi", "pi", "lp"],
        default="pi",
        help="Algorithm used to generate the path",
    )
    verify_parser.add_argument(
        "--grid",
        nargs="+",
        default=[f"data/mdp/grids/grid{i}.txt" for i in range(10, 110, 10)],
        help="List of grid files to verify (default: grid10 to grid100)",
    )
    verify_parser.add_argument("--keep-tmp", action="store_true")
    verify_parser.set_defaults(func=lambda args: run_mdp_verify(sys.argv[2:]))

    # Visualize MDP solution path on a maze grid
    viz_parser = subparsers.add_parser(
        "visualize_mdp", help="Visualize a maze grid with or without a path"
    )
    viz_parser.add_argument("--grid_file", required=True, help="Path to the grid file")
    viz_parser.add_argument("--path_file", help="Path to the path file (optional)")
    viz_parser.add_argument("--output_file", required=True, help="Output image path")
    viz_parser.set_defaults(func=lambda args: run_mdp_visualize(sys.argv[2:]))


    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
