import argparse
import subprocess
import sys

def run_bandits(args):
    subprocess.run(["python", "src/bandits/bandit.py"] + args)

def run_mdp(args):
    subprocess.run(["python", "src/mdp/planner.py"] + args)

def run_windy_gridworld(args):
    subprocess.run(["python", "src/windy_gridworld/gridworld.py"] + args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Agents Projects Runner")
    subparsers = parser.add_subparsers(dest='command')

    bandit_parser = subparsers.add_parser('bandits', help='Run Bandits experiments')
    bandit_parser.add_argument('--instance', required=True)
    bandit_parser.add_argument('--algorithm', required=True)
    bandit_parser.add_argument('--horizon', required=True)
    bandit_parser.add_argument('--randomSeed', default='0')

    mdp_parser = subparsers.add_parser('mdp', help='Run MDP planning')
    mdp_parser.add_argument('--mdp', required=True)
    mdp_parser.add_argument('--algorithm', required=True, choices=['vi', 'pi', 'lp'])
    mdp_parser.add_argument('--policy', required=True)

    gridworld_parser = subparsers.add_parser('windy', help='Run Windy Gridworld experiments')
    gridworld_parser.add_argument('--episodes', type=int, default=170)

    args, unknown_args = parser.parse_known_args()

    if args.command == 'bandits':
        run_bandits(sys.argv[2:])
    elif args.command == 'mdp':
        run_mdp(sys.argv[2:])
    elif args.command == 'windy':
        run_windy_gridworld(sys.argv[2:])
    else:
        parser.print_help()
