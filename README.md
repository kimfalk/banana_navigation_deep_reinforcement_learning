# Udacity DRL nanodegree - Project 1: Navigation
In this project, we train an agent to navigate, and collect bananas in a 
arge, square world, provided by a Unity machine learning agent environment. 
More information on the Unity ml-agents can be found 
[here](https://github.com/Unity-Technologies/ml-agents).

## Project Details
The agent interfaces to an environment which is characterised as follows:

A reward of +1 is provided for collecting a yellow banana, and a reward of 
-1 is provided for collecting a blue banana. Thus, the goal of your agent 
is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, 
along with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. Four discrete actions are available, corresponding to:
* 0 - move forrward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Repository structure
The code is structured as follows: 
* **Navigation.ipynb**: this is where the deep rl agent is trained.
* **dgn_agent.py**: this module implements a class to represent a vanilla dqn agent.
* **model.py**: this module contains the implementation of the neural network approximating the action value function.
* **replay_memory.py**: this module contains the replay memory. 
* **checkpoint.pth**: this is the binary containing the trained neural network weights.

### Dependencies
* python 3.6
* numpy: install with 'pip install numpy'.
* PyTorch: install by following the instructions [here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows).
* ml-agents: install by following instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).

## Getting Started
In cell 2 of Navigation.ipynb we import the Unity environment from a remote server. For a local installation of the 
Unity ml-agents, please refer to the following two sources:
* [Linux, Mac](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* [Windows 10](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md)

## Instructions
This is a jupyter notebook project. To run the code and train the deep reinforcement learning agent, you simply 
execute each of the cells in **Navigation.ipynb**. After training, the average score per hundred episodes will be displayed.