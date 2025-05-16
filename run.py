import argparse
import subprocess
import sys
import logging
from dataclasses import dataclass, field
from typing import List

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BanditConfig:
    instance: str
    algorithm: str
    rseed: int = 0
    epsilon: float = None
    horizon: int = 1000


@dataclass
class WindyGridworldConfig:
    episodes: int = 170
    epsilon: float = 0.1
    discount: float = 1.0
    learning_rate: float = 0.4

@dataclass
class MDPGenerateConfig:
    output_file: str
    num_states: int = 10
    num_actions: int = 5
    gamma: float = 0.95
    mdptype: str = "episodic"
    rseed: int = 42


@dataclass
class MDPSolveConfig:
    grid: str
    algorithm: str = "pi"
    keep_tmp: bool = False


@dataclass
class MDPVerifyConfig:
    algorithm: str = "pi"
    grid: List[str] = field(
        default_factory=lambda: [
            f"data/mdp/grids/grid{i}.txt" for i in range(10, 110, 10)
        ]
    )
    keep_tmp: bool = False


@dataclass
class MDPVisualizeConfig:
    grid_file: str
    output_file: str
    path_file: str = None


class SubcommandRunner:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def _log(self, level, message):
        if self.verbose:
            logger.log(level, message)

    def run(self, args: List[str]):
        raise NotImplementedError


class BanditRunner(SubcommandRunner):
    def run(self, config: BanditConfig):
        command = [
            "python",
            "src/bandits/main.py",
            f"--instance={config.instance}",
            f"--algorithm={config.algorithm}",
            f"--rseed={config.rseed}",
        ]
        if config.epsilon is not None:
            command.append(f"--epsilon={config.epsilon}")
        command.append(f"--horizon={config.horizon}")
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class WindyGridworldRunner(SubcommandRunner):
    def run(self, config: WindyGridworldConfig):
        command = [
            "python",
            "src/windy_gridworld/main.py",
            f"--episodes={config.episodes}",
            f"--epsilon={config.epsilon}",
            f"--discount={config.discount}",
            f"--learning-rate={config.learning_rate}",
        ]
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class MDPGenerateRunner(SubcommandRunner):
    def run(self, config: MDPGenerateConfig):
        command = [
            "python",
            "src/mdp/generate.py",
            f"--num_states={config.num_states}",
            f"--num_actions={config.num_actions}",
            f"--gamma={config.gamma}",
            f"--mdptype={config.mdptype}",
            f"--rseed={config.rseed}",
            f"--output_file={config.output_file}",
        ]
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class MDPSolveRunner(SubcommandRunner):
    def run(self, config: MDPSolveConfig):
        command = [
            "python",
            "src/mdp/main.py",
            f"--grid={config.grid}",
            f"--algorithm={config.algorithm}",
        ]
        if config.keep_tmp:
            command.append("--keep-tmp")
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class MDPVerifyRunner(SubcommandRunner):
    def run(self, config: MDPVerifyConfig):
        command = ["python", "src/mdp/verify.py", f"--algorithm={config.algorithm}"]
        command.append("--grid")
        command.extend(config.grid)
        if config.keep_tmp:
            command.append("--keep-tmp")
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


class MDPVisualizeRunner(SubcommandRunner):
    def run(self, config: MDPVisualizeConfig):
        command = [
            "python",
            "src/mdp/visualize.py",
            f"--grid_file={config.grid_file}",
            f"--output_file={config.output_file}",
        ]
        if config.path_file:
            command.append(f"--path_file={config.path_file}")
        self._execute(command)

    def _execute(self, command: List[str]):
        self._log(logging.DEBUG, f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self._log(
            logging.DEBUG, f"Command finished with exit code: {result.returncode}"
        )
        if self.verbose and result.stdout:
            print(result.stdout)
        if self.verbose and result.stderr:
            print(result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Agents Project Runner")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    subparsers = parser.add_subparsers(dest="command")

    # Bandits
    bandit_parser = subparsers.add_parser("bandits", help="Run Bandits experiments")
    bandit_parser.add_argument("--instance", default="data/bandits/instances/i-1.txt")
    bandit_parser.add_argument(
        "--algorithm",
        choices=[
            "ucb",
            "kl-ucb",
            "epsilon-greedy",
            "thompson-sampling",
            "thompson-sampling-with-hint",
        ],
        default="ucb",
    )
    bandit_parser.add_argument("--rseed", type=int, default=0)
    bandit_parser.add_argument("--epsilon", type=float, default=0.1)
    bandit_parser.add_argument("--horizon", type=int, default=1000)
    bandit_parser.set_defaults(runner=BanditRunner)

    # Windy Gridworld
    gridworld_parser = subparsers.add_parser(
        "windy", help="Run Windy Gridworld experiments"
    )
    gridworld_parser.add_argument("--episodes", type=int, default=170)
    gridworld_parser.add_argument("--epsilon", type=float, default=0.1)
    gridworld_parser.add_argument("--discount", type=float, default=1.0)
    gridworld_parser.add_argument("--learning-rate", type=float, default=0.4)
    gridworld_parser.set_defaults(runner=WindyGridworldRunner)

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
    gen_parser.set_defaults(runner=MDPGenerateRunner)

    # Solve MDP Maze
    solve_parser = subparsers.add_parser(
        "solve_mdp", help="Solve a maze-based MDP using vi, lp, or pi"
    )
    solve_parser.add_argument("--grid", required=True)
    solve_parser.add_argument("--algorithm", choices=["vi", "pi", "lp"], default="pi")
    solve_parser.add_argument("--keep-tmp", action="store_true")
    solve_parser.set_defaults(runner=MDPSolveRunner)

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
        nargs="*",
        default=[f"data/mdp/grids/grid{i}.txt" for i in range(10, 110, 10)],
        help="List of grid files to verify (default: grid10 to grid100)",
    )
    verify_parser.add_argument("--keep-tmp", action="store_true")
    verify_parser.set_defaults(runner=MDPVerifyRunner)

    # Visualize MDP solution path on a maze grid
    viz_parser = subparsers.add_parser(
        "visualize_mdp", help="Visualize a maze grid with or without a path"
    )
    viz_parser.add_argument("--grid_file", required=True, help="Path to the grid file")
    viz_parser.add_argument("--path_file", help="Path to the path file (optional)")
    viz_parser.add_argument("--output_file", required=True, help="Output image path")
    viz_parser.set_defaults(runner=MDPVisualizeRunner)

    args = parser.parse_args()

    verbose = args.verbose

    if hasattr(args, "runner"):
        runner_class = args.runner
        runner = runner_class(verbose)
        # Create a configuration object based on the parsed arguments
        config = None
        if args.command == "bandits":
            config = BanditConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in BanditConfig.__annotations__
                }
            )
        elif args.command == "windy":
            config = WindyGridworldConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in WindyGridworldConfig.__annotations__
                }
            )
        elif args.command == "generate_mdp":
            config = MDPGenerateConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in MDPGenerateConfig.__annotations__
                }
            )
        elif args.command == "solve_mdp":
            config = MDPSolveConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in MDPSolveConfig.__annotations__
                }
            )
        elif args.command == "verify_mdp":
            config = MDPVerifyConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in MDPVerifyConfig.__annotations__
                }
            )
        elif args.command == "visualize_mdp":
            config = MDPVisualizeConfig(
                **{
                    k: v
                    for k, v in vars(args).items()
                    if k in MDPVisualizeConfig.__annotations__
                }
            )

        if config:
            runner.run(config)
        else:
            print("No runner configured for this command.")
            parser.print_help()
    else:
        parser.print_help()
