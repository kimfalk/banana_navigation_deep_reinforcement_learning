
# Udacity DRL nanodegree - Project 1: Navigation

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

In this project, we train an agent to navigate and collect bananas in a square world. Provided by a Unity agent environment. 
More information on the Unity ml-agents can be found 
[here](https://github.com/Unity-Technologies/ml-agents).

## Project Details
The agent interface to an environment is as follows:

The agent receives a reward of +1 when collecting a yellow banana, and a reward of -1 when collecting a blue banana. Thus, the goal of the agent 
is to collect as many yellow bananas as possible while avoiding blue bananas.

The state-space has 37 dimensions and contains the agent's velocity, 
along with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. Four discrete actions are available, corresponding to:
* 0 - move forward.
* 1 - move backwards.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Repository structure
The code is structured as follows: 
* **Navigation.ipynb**: this notebook simulates the agent in the environment and trains the agent.
* **dgn_agent.py**: implements a double DQN agent.
* **model.py**:  contains the implementation of the neural network approximating the action-value function.
* **replay_memory.py**: contains the replay memory. 
* **checkpoint.pth**: this is the binary containing the trained neural network weights.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training the agent!  
