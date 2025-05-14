
# Intelligent Agents CS747

A collection of reinforcement learning and intelligent agents projects showcasing implementations of key algorithms and their comparative analysis. This repository is structured for clarity, reproducibility, and ease of demonstration. These projects were developed as part of the *CS747: Foundations of Intelligent and Learning Agents* course at IIT Bombay.

## Projects Included:

- **Bandits** ([src/bandits](src/bandits)): Implementation and comparison of classic multi-armed bandit algorithms: ε-greedy, UCB, KL-UCB, and Thompson Sampling. Includes an extension based on "Thompson sampling with a hint" using a permutation of true means. See the [report](src/bandits/submission/Report.pdf) for algorithmic insights and evaluation results.

- **MDP Maze Solver** ([src/mdp](src/mdp)): Solves mazes by modeling them as Markov Decision Processes using Value Iteration, Linear Programming, and Howard's Policy Iteration. Includes custom maze encoding/decoding, verification scripts, and visualizations. Refer to the [report](src/mdp/submission/Report.pdf) for implementation details and outcomes.

- **Windy Gridworld** ([src/windy_gridworld](src/windy_gridworld)): Uses RL algorithms like Sarsa (4-directional and King’s moves), Q-Learning, and Expected Sarsa to solve the Windy Gridworld problem under various wind conditions. Includes visualizations of policy paths and algorithm comparison plots. See [report](src/windy_gridworld/submission/Report.pdf) for findings.

## Quick Setup

Clone this repo:
```bash
git clone https://github.com/yourusername/Intelligent-Agents-CS747.git
cd Intelligent-Agents-CS747
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Bandits
Run experiments comparing multi-armed bandit algorithms:
```bash
python run.py bandits --instance instances/i-1.txt --algorithm ucb --horizon 1000 --epsilon 0.1
```

### MDP Maze Solver
Solve an MDP maze using a specified algorithm:
```bash
python run.py mdp --mdp data/mdp-10-5.txt --algorithm vi --policy output/policy.txt
```

### Windy Gridworld
Run reinforcement learning algorithms on Windy Gridworld:
```bash
python run.py windy --episodes 170
```

## Project Structure
```
Intelligent-Agents-CS747/
├── src/
│   ├── bandits/
│   │   └── submission/
│   ├── mdp/
│   │   └── submission/
│   └── windy_gridworld/
│       └── submission/
├── run.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Each `submission/` folder contains the main code, data generators, verification scripts, references, and output plots.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
