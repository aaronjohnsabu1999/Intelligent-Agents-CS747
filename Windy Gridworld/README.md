# The Windy Gridworld Problem
## A Tale of Fighting the Elements...

This directory contains files pertaining to Assignment 03 of the course on the 'Foundations of Intelligent and Learning Agents' (CS 747) at IIT Bombay. The problem statement involves solving the famous Windy Gridworld problem using the *Sarsa* algorithm for normal (4) moves as well as King's (8) moves for non-stochastic and stochastic wind values, as well as using the *Q-Learning* and *Extended Sarsa* algorithms with normal moves.  

A detailed [report](./Report.pdf) on the methods used for the implementation and comparison of the algorithms has been provided.  
[submission](./submission) contains the [main code](./submission/gridworld.py), as well as the [references](./submission/references.txt) used for developing the algorithms and the [plots](./submission/plots) of the comparison of the algorithms and of the optimal paths. Running gridworld.py will store the comparison plot in the folder along with displaying the optimal paths for each algorithm (excluding the stochastic wind case).