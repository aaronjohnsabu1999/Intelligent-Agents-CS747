import argparse
import subprocess
import sys

def run_bandits(args):
    subprocess.run(["python", "src/bandits/main.py"] + args)

def run_mdp_generate(args):
    subprocess.run(["python", "src/mdp/main.py", "generate"] + args)

def run_mdp_solve(args):
    subprocess.run(["python", "src/mdp/main.py", "solve"] + args)

def run_windy_gridworld(args):
    subprocess.run(["python", "src/windy_gridworld/main.py"] + args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Agents Project Runner")
    subparsers = parser.add_subparsers(dest='command')

    # Bandits
    bandit_parser = subparsers.add_parser('bandits', help='Run Bandits experiments')
    bandit_parser.add_argument('--instance', required=True)
    bandit_parser.add_argument('--algorithm', required=True)
    bandit_parser.add_argument('--horizon', required=True)
    bandit_parser.add_argument('--randomSeed', default='0')
    bandit_parser.set_defaults(func=lambda args: run_bandits(sys.argv[2:]))

    # Windy Gridworld
    gridworld_parser = subparsers.add_parser('windy', help='Run Windy Gridworld experiments')
    gridworld_parser.add_argument('--episodes', type=int, default=170)
    gridworld_parser.set_defaults(func=lambda args: run_windy_gridworld(sys.argv[2:]))

    # Generate MDP
    gen_parser = subparsers.add_parser('generate_mdp', help='Generate a random MDP instance')
    gen_parser.add_argument('--S', type=int, default=5)
    gen_parser.add_argument('--A', type=int, default=2)
    gen_parser.add_argument('--gamma', type=float, default=0.9)
    gen_parser.add_argument('--mdptype', choices=["continuing", "episodic"], default="continuing")
    gen_parser.add_argument('--rseed', type=int, default=0)
    gen_parser.add_argument('--mdp_output_path', required=True)
    gen_parser.set_defaults(func=lambda args: run_mdp_generate(sys.argv[2:]))

    # Solve MDP
    solve_parser = subparsers.add_parser('solve_mdp', help='Solve an MDP using vi, lp, or pi')
    solve_parser.add_argument('--mdp_input_path', required=True)
    solve_parser.add_argument('--algorithm', required=True, choices=['vi', 'lp', 'pi'])
    solve_parser.add_argument('--policy_output_path')
    solve_parser.add_argument('--value_output_path')
    solve_parser.set_defaults(func=lambda args: run_mdp_solve(sys.argv[2:]))

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
