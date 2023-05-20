import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

from env.driver import driver_env
n_drivers = 1000
train_mode = True
env = driver_env()

print(f'Enviroment Reset: {env.reset()}')

from dqn.dqn_agent import Agent

agent = Agent(state_size=2, action_size=1, seed=42)

import torch
def train(n_episodes=1e5, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, int(n_episodes)+1):
        state = env.reset(train_mode=True) # reset the environment and get state
        
        action = agent.act(np.array(state), eps) #Select an Action
        action = ((action + 1) / (1 + 1)) * 30
        env_info = env.step(action)       # send the action to the environment
        for entry in env_info:
            reward = entry[0]
            next_state = entry[1]
            done = entry[2]
            agent.step(state, action, reward, next_state, done)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return

train()
