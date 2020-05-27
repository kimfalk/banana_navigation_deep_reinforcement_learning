# Navigation 

Your agent is running around in a squared walled world where blue and yellow bananas are scattered around
we need to teach the agent to pick the yellow bananas which it will give it a reward of +1 while avoid 
picking up the blue bananas which will give a reward of -1. 

In the following I will describing how I implemented learning algorithm, and explain what I would do next
given the time.

The solution used double-DQN agent, with Prioritized Experience Replay.

## Implementation

The main algorithm used is *DDQN* - Double Deep Q-learning Network. This algorithm uses function 
approximation algorithm. 

The Q-learning function is approximated using a neural network. The agen will query the model for every
step, which will return an appropriate action. 

We use what the agent experience to improve the model. To avoid having the model running after their own 
tail, we use two identical models. One is the one used online to guide the agent, and one that is used 
during training of the neural networks. This is done to keep the target constant for small periods of the
time, enabling the agent to improve more steadily. 

For each step the agent saves the whole experience, of the step. Where it was, where it went, the action 
taken, reward, whether it is done. As with all training of ML algorithms its a good idea not to train 
the model with the data represented chronology. We therefore save the experiences and then sample from
the buffer.


These samples can be uniform, but I think 
## Ideas for future work 
