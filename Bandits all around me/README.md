# bandits-all-around-me
## A Comparison of Multi-Armed Bandit Algorithms

This directory contains files pertaining to Assignment 01 of the course on the 'Foundations of Intelligent and Learning Agents' (CS 747) at IIT Bombay. The problem statement involves the implementation and comparison of the *epsilon-greedy*, *UCB*, *KL-UCB*, and *Thompson sampling* algorithms, as well as an open question of implementing an algorithm with lower regret while incorporating the principle of *Thompson sampling with a hint* that a random (maybe sorted) permutation of the true means is provided.

A detailed [report](./Report.pdf) on the techniques utilized throughout the assignment has been provided about the methods used for the implementation and comparison of the algorithms.  
Three test instances have been provided in [instances](./instances) of 2, 5, and 25 arms respectively.  
[submission](./submission) contains the [main code](./submission/bandit.py), as well as data generation scripts `t***run.sh`, [references](./submission/references.txt), an [error-checking script](./submission/check.sh), an [output specification file](./submission/outputFormat.txt), [output files](./submission/outputs) and [plots](./submission/outputs/plots)
