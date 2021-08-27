# Reasoning about agent's beliefs about their goals  
This repo implements a generative model of goal-directed actions in an environment with goals of varying attributes. With this model, we aim to investigate how people draw inferences of someone else's beliefs about the nature of their goals from the order in which they pursue their goials

Details of the game are as follows:
An agent is placed in an environment with goals of different attributes. Some requires the completion of other goals while others have rewards that diminishes with time. The agent's objective is to collect goals in a manner that will allows her to obtain the maximum reward. 

## Implementation Details

Our environment is an adaptation of DeepMind's Sequential Social Dilemma (SSD) multi-agent game-theoretic environments [[1]](https://arxiv.org/abs/1702.03037)
Our model takes inspiration from the Naive Utility Calculus framework of Action Understanding [[2]](https://www.sciencedirect.com/science/article/abs/pii/S0010028520300633)


## Relevant papers

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2. Jara-Ettinger, J., Schulz, L. E., & Tenenbaum, J. B. (2020). The naive utility calculus as a unified, quantitative framework for action understanding. Cognitive Psychology, 123, 101334.

