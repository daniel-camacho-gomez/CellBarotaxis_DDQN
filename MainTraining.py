import gymnasium as gym
import gym
from gym import spaces   
import numpy as np 
import math
import statistics
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import copy
import os
import time   
import pickle 

import optuna 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



from CellMigrationEnv import CellMigrationEnv
from DQN import DQN
from plotting_options import plot_reward_evolution, cell_trajectory_pressure

# Reduce TSIM in CellMigrationEnv for training. 
geometry = ["straight","top","bot"]

# Register the custom environment
gym.envs.register(
    id="CellMigration-v0",
    entry_point='CellMigrationEnv:CellMigrationEnv',
    kwargs={'N_obs': 72, 'geometry':geometry}
)


# Create an instance of the custom environment
env = gym.make('CellMigration-v0')

# This is to avoid the error when plotting because of OpenMP 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)}h {int(minutes)}m {int(seconds)}s'



# Enable interactive mode
plt.ion()

# ------------------------------------------------------------------------------
# Hyperparameters and utilities
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE  = 512 #512
GAMMA       = 0.1 #0.9
EPS_START   = 0.9
EPS_END     = .0001
EPS_DECAY   = 1000
TAU         = 0.005
LR          = 1e-4
# ------------------------------------------------------------------------------

# Get number of actions from gym action space
n_actions      = env.action_space.n
# Get the number of state observations
n_observations = env.N_obs

# Hidden layer 
N_HL = [100, 100, 100]

init_method = 'default' 

policy_net      = DQN(n_observations, N_HL, n_actions, init_method).to(device)
target_net      = DQN(n_observations, N_HL, n_actions, init_method).to(device)
best_policy_net = DQN(n_observations, N_HL, n_actions, init_method).to(device)
best_target_net = DQN(n_observations, N_HL, n_actions, init_method).to(device)
target_net.load_state_dict(policy_net.state_dict())
best_policy_net.load_state_dict(policy_net.state_dict())
best_target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

memory    = ReplayMemory(10000)


steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) 
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
        


    
num_episodes = 100000
policy_net.train()
target_net.train()

reward         = {}
reward_history = []  # Initialize a list to store rewards over episodes
start_time     = time.time()  # Record the start time
for i_episode in range(num_episodes):
    for geom in env.geom:
        # Initialize the environment and get its state
        env.geometry_selection(geom)
        state,_     = env.reset()
        state     = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward[geom], terminated, obs_pressure, _ = env.step(action.item())
            reward_tensor = torch.tensor([reward[geom]], device=device)
            done = terminated
        
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward_tensor)

            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization (on the policy network)
            optimize_model()
        
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
      
   
    # scheduler.step()
        
    # Append the current reward to the history
    reward_history.append(copy.copy(reward))
    
    # Calculate the mean of values in all dictionaries in reward_history
    mean_rewards = [statistics.mean(d.values()) for d in reward_history]
    # Save the best net so far 
    if i_episode>0 and statistics.mean(reward_history[-1].values()) > max(mean_rewards[:-1]):
        best_policy_net.load_state_dict(policy_net.state_dict())
        best_target_net.load_state_dict(target_net.state_dict())
    
    
    if i_episode % 50 == 0:
        plot_reward_evolution(reward_history)
        plt.pause(0.01)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time



    print('------------------------------------------------------------------------')
    print('            Episode: ' +str(i_episode))
    rounded_rewards_hist = {key: round(value, 4) for key, value in reward_history[-1].items()}
    print('     Current reward: ' +str(rounded_rewards_hist))
    print('Current mean reward: ' +str(round(statistics.mean(reward_history[-1].values()),4)))
    print('   Best mean reward: ' + str(round(max(mean_rewards),4)) + ' in episode ' + str(np.argmax(mean_rewards)))
    print('              Stop?: ' + str(max(mean_rewards) == 1))
    print('       Elapsed time: ' + format_time(elapsed_time))
    print('------------------------------------------------------------------------')


# Disable interactive mode after the loop
plt.ioff()

# Show the final plot (optional)
plt.show()         


print('Training Done')

