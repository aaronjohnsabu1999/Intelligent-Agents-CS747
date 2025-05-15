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
    success: int = 0
    failure: int = 0
    estimated_prob: float = 0.0


class BanditExperiment:
    algos = [
        "epsilon-greedy",
        "ucb",
        "kl-ucb",
        "thompson-sampling",
        "thompson-sampling-with-hint",
    ]

    def __init__(self, instance: str, algorithm: str, rseed: int, epsilon: float, horizon: int):
        self.instance = instance
        self.algorithm = algorithm
        self.rseed = rseed
        self.epsilon = epsilon
        self.horizon = horizon
        self.arms: List[float] = self.load_instance()

    def load_instance(self) -> List[float]:
        with open(self.instance) as f:
            return [float(line.strip()) for line in f.readlines()]

    def run(self) -> float:
        random.seed(self.rseed)

        if self.algorithm == "epsilon-greedy":
            picks = self.epsilon_greedy()
        elif self.algorithm == "ucb":
            picks = self.ucb(kl=False)
        elif self.algorithm == "kl-ucb":
            picks = self.ucb(kl=True)
        elif self.algorithm == "thompson-sampling":
            picks = self.thompson()
        elif self.algorithm == "thompson-sampling-with-hint":
            picks = self.thompson_hint()
        else:
            raise ValueError("Unknown algorithm")

        regret = self.calculate_regret(picks)
        logger.info(self.format_result(regret))
        return regret

    def format_result(self, regret: float) -> str:
        return f"{self.instance}, {self.algorithm}, {self.rseed}, {self.epsilon}, {self.horizon}, {regret}"

    def epsilon_greedy(self) -> Dict[float, ArmStats]:
        picks = {arm: ArmStats() for arm in self.arms}
        for _ in range(self.horizon):
            pick = (
                random.choice(range(len(self.arms)))
                if random.random_sample() < self.epsilon
                else np.argmax([np.mean(p.rewards) if p.rewards else 0 for p in picks.values()])
            )
            arm = self.arms[pick]
            reward = int(random.random_sample() < arm)
            stats = picks[arm]
            stats.pulls += 1
            stats.rewards.append(reward)
        return picks

    def ucb(self, kl: bool) -> Dict[float, ArmStats]:
        def log(x): return math.log(x) if x > 0 else 0

        def ucb_calc(p: float, n: int, t: int) -> float:
            if not kl:
                return p + math.sqrt(2 * log(t) / n)
            start, end, c, tol = p, 1.0, 3, 1e-4
            final = (log(t) + c * log(log(t))) / n
            while abs(start - end) > tol:
                mid = (start + end) / 2
                kl_val = p * log(p / mid) + (1 - p) * log((1 - p) / (1 - mid))
                start, end = (mid, end) if kl_val <= final else (start, mid)
            return (start + end) / 2

        picks = {arm: ArmStats(pulls=1, rewards=[int(random.random_sample() < arm)]) for arm in self.arms}
        for t in range(len(picks), self.horizon):
            for stats in picks.values():
                stats.estimated_prob = np.mean(stats.rewards)
            scores = [ucb_calc(p.estimated_prob, p.pulls, t) for p in picks.values()]
            arm = self.arms[np.argmax(scores)]
            reward = int(random.random_sample() < arm)
            stats = picks[arm]
            stats.pulls += 1
            stats.rewards.append(reward)
        return picks

    def thompson(self) -> Dict[float, ArmStats]:
        picks = {arm: ArmStats() for arm in self.arms}
        for _ in range(self.horizon):
            betas = [random.beta(p.success + 1, p.failure + 1) for p in picks.values()]
            arm = self.arms[np.argmax(betas)]
            reward = int(random.random_sample() < arm)
            stats = picks[arm]
            stats.pulls += 1
            stats.rewards.append(reward)
            stats.success += reward
            stats.failure += 1 - reward
            stats.estimated_prob = stats.success / (stats.success + stats.failure)
        return picks

    def thompson_hint(self) -> Dict[float, ArmStats]:
        picks = {arm: ArmStats() for arm in self.arms}
        hints = np.sort(self.arms)
        for t in range(self.horizon):
            if t < 0.2 * self.horizon or random.random_sample() < 0.3:
                betas = [random.beta(p.success + 1, p.failure + 1) for p in picks.values()]
                arm = self.arms[np.argmax(betas)]
            else:
                true_max = max(hints)
                phi = 1e4
                distr = random.beta(true_max * phi, (1 - true_max) * phi)
                emps = [abs(p.estimated_prob - distr) for p in picks.values()]
                arm = self.arms[np.argmin(emps)]
            reward = int(random.random_sample() < arm)
            stats = picks[arm]
            stats.pulls += 1
            stats.rewards.append(reward)
            stats.success += reward
            stats.failure += 1 - reward
            stats.estimated_prob = stats.success / (stats.success + stats.failure)
        return picks

    def calculate_regret(self, picks: Dict[float, ArmStats]) -> float:
        opt_arm = max(self.arms)
        total = self.horizon * opt_arm
        reward = sum(sum(p.rewards) for p in picks.values())
        return round(total - reward, 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="data/bandits/instances/i-1.txt")
    parser.add_argument("--algorithm", default="ucb", choices=BanditExperiment.algos)
    parser.add_argument("--rseed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=1000)
    args = parser.parse_args()

    BanditExperiment(
        args.instance, args.algorithm, args.rseed, args.epsilon, args.horizon
    ).run()