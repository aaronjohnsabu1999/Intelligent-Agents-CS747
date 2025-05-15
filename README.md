# Intelligent Agents

A collection of reinforcement learning and intelligent agents projects showcasing implementations of key algorithms and their comparative analysis. These projects were developed as part of the *CS747: Foundations of Intelligent and Learning Agents* course at IIT Bombay.

## Projects Included

### Bandits (`src/bandits`)

Implements and compares classical multi-armed bandit algorithms: **ε-greedy**, **UCB**, **KL-UCB**, and **Thompson Sampling**, along with a custom variant: **Thompson Sampling with a hint**. Each algorithm minimizes cumulative regret across different horizons and random seeds.

**Key Findings:**
- Thompson Sampling generally outperforms others in regret minimization.
- KL-UCB improves over UCB using a tighter confidence bound via binary search.
- ε-Greedy performs best at `ε ≈ 0.02` — striking a balance between exploration and exploitation.
- The Thompson Sampling "hinted" version leverages knowledge of true means, improving early performance through a custom Beta-distribution-based selector.

Includes regret plots over multiple seeds and horizons, as well as parameter studies.

---

### MDP Maze Solver (`src/mdp`)

Solves mazes by modeling them as Markov Decision Processes using:
- **Value Iteration**
- **Linear Programming** (via PuLP)
- **Howard’s Policy Iteration**

**Pipeline:**
1. `encoder.py` transforms grid mazes into MDPs.
2. `solver.py` computes optimal policy.
3. `decoder.py` reconstructs the shortest path using the policy.

**Insights:**
- LP is consistently fastest for large mazes.
- Howard's Policy Iteration performs well on small problems but becomes costly as maze complexity grows.
- Visual comparisons confirm that solved mazes follow intuitive paths with minimal steps.

Benchmarks for runtime across methods and visualizations for grid navigation are included.

### Windy Gridworld (`src/windy_gridworld`)

Adopts the Sutton & Barto Windy Gridworld challenge with multiple RL approaches:
- **Sarsa** (normal and King’s moves)
- **Sarsa with stochastic wind**
- **Q-Learning**
- **Expected Sarsa**

**Key Results:**
- Sarsa with King’s Moves converges fastest due to shorter episodes.
- Q-Learning and Expected Sarsa outperform standard Sarsa on stability and convergence.
- The stochastic wind variant adds realistic randomness but slows convergence.
- Paths from all agents are visualized for both deterministic and windy environments.

Gridworld is defined as an episodic MDP with reward shaping and stepwise convergence plotting.

---

## Quick Setup

```bash
git clone https://github.com/aaronjohnsabu1999/intelligent-agents.git
cd intelligent-agents
pip install -r requirements.txt
```

## How to Run

### Bandits
```bash
python run.py bandits --instance instances/i-1.txt --algorithm ucb --horizon 1000 --epsilon 0.1
```

### MDP Maze Solver
```bash
python run.py solve_mdp \
  --grid data/mdp/grids/grid10.txt \
  --algorithm pi
```
To create a synthetic MDP file using the generator script:
```bash
python run.py generate_mdp \
  --num_states 10 \
  --num_actions 5 \
  --gamma 0.95 \
  --mdptype episodic \
  --rseed 42 \
  --output_file src/mdp/tmp/generated_mdp.txt
```
To verify all grids:
```bash
python run.py verify_mdp --algorithm pi
```
Or to test just one grid:
```bash
python run.py verify_mdp --algorithm vi --grid data/maze/grid40.txt
```
To visualize an unsolved maze:
```bash
python run.py visualize_mdp \
  --grid_file data/mdp/grids/grid10.txt \
  --output_file plots/mdp/grid10_unsolved.png
```
To visualize a maze with path overlaid:
```bash
python run.py visualize_mdp \
  --grid_file data/mdp/grids/grid10.txt \
  --path_file data/mdp/paths/path10.txt \
  --output_file plots/mdp/grid10_solved.png
```

### Windy Gridworld
```bash
python run.py windy --episodes 170
```

## References

- [`./references/mdp_references.txt`](./references/mdp_references.txt) – Value Iteration, Policy Iteration, and Linear Programming.
- [`./references/bandits_references.txt`](./references/bandits_references.txt) – Multi-Armed Bandits and Exploration Strategies.
- [`./references/windy_gridworld_references.txt`](./references/windy_gridworld_references.txt) – Reinforcement Learning in Grid-based Environments.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.