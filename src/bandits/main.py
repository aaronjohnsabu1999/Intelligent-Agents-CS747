import sys
import math
import numpy as np
import argparse
import warnings
import logging
from numpy import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    pulls: int = 0
    rewards: List[int] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    estimated_probability: float = 0.0


class BanditExperiment:
    ALGORITHMS = [
        "epsilon-greedy",
        "ucb",
        "kl-ucb",
        "thompson-sampling",
        "thompson-sampling-with-hint",
    ]

    def __init__(
        self,
        instance_path: str,
        algorithm: str,
        seed: int,
        epsilon: float,
        horizon: int,
    ):
        self.instance_path = instance_path
        self.algorithm = algorithm
        self.seed = seed
        self.epsilon = epsilon
        self.horizon = horizon
        self.arm_probabilities: List[float] = self._load_instance()

    def _load_instance(self) -> List[float]:
        with open(self.instance_path) as f:
            return [float(line.strip()) for line in f.readlines()]

    def run(self) -> float:
        random.seed(self.seed)

        if self.algorithm == "epsilon-greedy":
            arm_stats = self._epsilon_greedy_strategy()
        elif self.algorithm == "ucb":
            arm_stats = self._ucb_strategy(use_kl=False)
        elif self.algorithm == "kl-ucb":
            arm_stats = self._ucb_strategy(use_kl=True)
        elif self.algorithm == "thompson-sampling":
            arm_stats = self._thompson_sampling_strategy()
        elif self.algorithm == "thompson-sampling-with-hint":
            arm_stats = self._thompson_sampling_with_hint_strategy()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        regret = self._calculate_regret(arm_stats)
        logger.info(self._format_result(regret))
        return regret

    def _format_result(self, regret: float) -> str:
        return f"{self.instance_path}, {self.algorithm}, {self.seed}, {self.epsilon}, {self.horizon}, {regret}"

    def _epsilon_greedy_strategy(self) -> Dict[float, ArmStats]:
        arm_stats = {arm: ArmStats() for arm in self.arm_probabilities}
        for _ in range(self.horizon):
            if random.random_sample() < self.epsilon:
                chosen_arm_index = random.choice(range(len(self.arm_probabilities)))
            else:
                mean_rewards = [
                    np.mean(stats.rewards) if stats.rewards else 0
                    for stats in arm_stats.values()
                ]
                chosen_arm_index = np.argmax(mean_rewards)

            chosen_arm_probability = self.arm_probabilities[chosen_arm_index]
            reward = 1 if random.random_sample() < chosen_arm_probability else 0
            stats = arm_stats[chosen_arm_probability]
            stats.pulls += 1
            stats.rewards.append(reward)
        return arm_stats

    def _ucb_strategy(self, use_kl: bool) -> Dict[float, ArmStats]:
        def kullback_leibler_divergence(p: float, q: float) -> float:
            return (
                p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
                if 0 < p < 1 and 0 < q < 1
                else float("inf")
            )

        def calculate_ucb(mean_reward: float, pulls: int, time_step: int) -> float:
            if pulls == 0:
                return float("inf")
            if not use_kl:
                return mean_reward + math.sqrt(2 * math.log(time_step) / pulls)
            low, high, tolerance = mean_reward, 1.0, 1e-4
            bound = (
                (math.log(time_step) + 3 * math.log(math.log(time_step))) / pulls
                if time_step > 1
                else 0
            )
            while high - low > tolerance:
                mid = (low + high) / 2
                if kullback_leibler_divergence(mean_reward, mid) <= bound:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2

        arm_stats = {
            arm: ArmStats(pulls=1, rewards=[1 if random.random_sample() < arm else 0])
            for arm in self.arm_probabilities
        }
        for t in range(len(arm_stats), self.horizon):
            for stats in arm_stats.values():
                stats.estimated_probability = np.mean(stats.rewards)
            ucb_values = [
                calculate_ucb(stats.estimated_probability, stats.pulls, t + 1)
                for stats in arm_stats.values()
            ]
            chosen_arm_index = np.argmax(ucb_values)
            chosen_arm_probability = self.arm_probabilities[chosen_arm_index]
            reward = 1 if random.random_sample() < chosen_arm_probability else 0
            stats = arm_stats[chosen_arm_probability]
            stats.pulls += 1
            stats.rewards.append(reward)
        return arm_stats

    def _thompson_sampling_strategy(self) -> Dict[float, ArmStats]:
        arm_stats = {arm: ArmStats() for arm in self.arm_probabilities}
        for _ in range(self.horizon):
            beta_samples = [
                random.beta(stats.successes + 1, stats.failures + 1)
                for stats in arm_stats.values()
            ]
            chosen_arm_index = np.argmax(beta_samples)
            chosen_arm_probability = self.arm_probabilities[chosen_arm_index]
            reward = 1 if random.random_sample() < chosen_arm_probability else 0
            stats = arm_stats[chosen_arm_probability]
            stats.pulls += 1
            stats.rewards.append(reward)
            stats.successes += reward
            stats.failures += 1 - reward
            stats.estimated_probability = stats.successes / (
                stats.successes + stats.failures
            )
        return arm_stats

    def _thompson_sampling_with_hint_strategy(self) -> Dict[float, ArmStats]:
        arm_stats = {arm: ArmStats() for arm in self.arm_probabilities}
        sorted_arm_probabilities = np.sort(self.arm_probabilities)
        for t in range(self.horizon):
            if t < 0.2 * self.horizon or random.random_sample() < 0.3:
                beta_samples = [
                    random.beta(stats.successes + 1, stats.failures + 1)
                    for stats in arm_stats.values()
                ]
                chosen_arm_index = np.argmax(beta_samples)
            else:
                true_max_prob = max(sorted_arm_probabilities)
                concentration = 1e4
                hint_distribution = random.beta(
                    true_max_prob * concentration, (1 - true_max_prob) * concentration
                )
                estimation_errors = [
                    abs(stats.estimated_probability - hint_distribution)
                    for stats in arm_stats.values()
                ]
                chosen_arm_index = np.argmin(estimation_errors)

            chosen_arm_probability = self.arm_probabilities[chosen_arm_index]
            reward = 1 if random.random_sample() < chosen_arm_probability else 0
            stats = arm_stats[chosen_arm_probability]
            stats.pulls += 1
            stats.rewards.append(reward)
            stats.successes += reward
            stats.failures += 1 - reward
            stats.estimated_probability = stats.successes / (
                stats.successes + stats.failures
            )
        return arm_stats

    def _calculate_regret(self, arm_stats: Dict[float, ArmStats]) -> float:
        optimal_arm_probability = max(self.arm_probabilities)
        total_optimal_reward = self.horizon * optimal_arm_probability
        total_actual_reward = sum(sum(stats.rewards) for stats in arm_stats.values())
        return round(total_optimal_reward - total_actual_reward, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-armed bandit experiment.")
    parser.add_argument(
        "--instance",
        default="data/bandits/instances/i-1.txt",
        help="Path to the bandit instance file.",
    )
    parser.add_argument(
        "--algorithm",
        default="ucb",
        choices=BanditExperiment.ALGORITHMS,
        help="The bandit algorithm to use.",
    )
    parser.add_argument(
        "--rseed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon parameter for epsilon-greedy.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Number of time steps in the experiment.",
    )
    args = parser.parse_args()

    BanditExperiment(
        args.instance, args.algorithm, args.rseed, args.epsilon, args.horizon
    ).run()
