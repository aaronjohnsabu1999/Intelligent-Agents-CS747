# MDPs-for-a-purpose
## A Tale of Solving Mazes using MDPs

This repository contains files pertaining to Assignment 02 of the course on the 'Foundations of Intelligent and Learning Agents' (CS 747) at IIT Bombay. The problem statement involves the implementation of the *value iteration*, *linear programming*, and *Howard's policy iteration* algorithms as they are, as well as using them to solve mazes with arbitrary start and end points.  

A detailed [report](./Report.pdf) on the techniques utilized throughout the assignment has been provided about the methods used for the implementation and comparison of the algorithms.  
[submission](./submission) contains the [main code](./submission/planner.py) for optimal policy and value function evaluation, the [encoder](./submission/encoder.py) for translating the maze into an MDP, and the [decoder](./submission/decoder.py) for realizing a human-readable solution from the optimal policy, as well as [references](./submission/references.txt), [planner verification](./submission/PlannerVerifyOutput.py) and [maze verification](./submission/MazeVerifyOutput.py) scripts, a [grid visualization](./submission/visualize.py) script, a [custom MDP generation](./submission/generateMDP.py) script and [plots](./submission/plots) of both unsolved and solved mazes.
