#! /usr/bin/python
import random
import argparse
import sys
import os
import numpy as np


class MDPGenerator:
    def __init__(self, num_states, num_actions, gamma=0.9, mdptype="continuing", rseed=0):
        self.S = num_states
        self.A = num_actions
        self.gamma = gamma
        self.mdptype = mdptype
        self.rng = random.Random(rseed)

    def generate(self):
        return (
            self._generate_continuing()
            if self.mdptype == "continuing"
            else self._generate_episodic()
        )

    def _generate_common_header(self):
        return [f"numStates {self.S}", f"numActions {self.A}", "start 0"]

    def _generate_continuing(self):
        lines = self._generate_common_header()
        lines.append("end -1")
        for s in range(self.S):
            for a in range(self.A):
                degree = self.rng.randint(1, min(5, self.S))
                next_states = self.rng.sample(range(self.S), degree)
                rewards = [self.rng.uniform(-1, 1) for _ in range(degree)]
                probs = [self.rng.random() for _ in range(degree)]
                total = sum(probs)
                for i in range(degree):
                    lines.append(
                        f"transition {s} {a} {next_states[i]} {rewards[i]} {probs[i] / total}"
                    )
        lines.append(f"mdptype {self.mdptype}")
        lines.append(f"discount {self.gamma}")
        return lines

    def _generate_episodic(self):
        lines = self._generate_common_header()
        ends = (
            self.rng.sample(range(self.S), min(2, self.S - 2))
            if self.S <= 5
            else self.rng.sample(range(self.S), max(2, self.S // 10))
        )
        path = [s for s in range(self.S) if s not in ends]
        self.rng.shuffle(path)
        ends.append(path[-1])
        lines.append("end " + " ".join(map(str, ends)))

        for s in range(self.S):
            for a in range(self.A):
                if s in ends:
                    continue
                next_state = path[path.index(s) + 1] if s in path[:-1] else s
                distractors = [x for x in range(self.S) if x != next_state]
                self.rng.shuffle(distractors)
                degree = self.rng.randint(1, min(5, self.S))
                rewards = [self.rng.uniform(-1, 1) for _ in range(degree)]
                weights = [self.rng.randint(1, 1000) for _ in range(degree)]
                total = sum(weights)
                for i in range(degree - 1):
                    lines.append(
                        f"transition {s} {a} {distractors[i]} {rewards[i]} {weights[i] / total}"
                    )
                lines.append(
                    f"transition {s} {a} {next_state} {rewards[-1]} {weights[-1] / total}"
                )
        lines.append(f"mdptype {self.mdptype}")
        lines.append(f"discount {self.gamma}")
        return lines

    def save_to_file(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self.generate()) + "\n")
