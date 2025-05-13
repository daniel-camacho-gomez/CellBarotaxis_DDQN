import gymnasium as gym
import gym
from gym import spaces    
import math
import statistics
import random
import matplotlib
import matplotlib.pyplot as pltv
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

from importlib import reload

from CellMigrationEnv import CellMigrationEnv
from DQN import DQN
from plotting_options import*

# Select the geometry
# geometry = ["straight"]
geometry = ["deadend"]
# geometry = ["twisted"]
# geometry = ["curved"]


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


# Load the policy_net from the saved file 
path = r'Data\TrainedNN.pth'
best_net = torch.load(path, map_location=device)
best_net.eval()
    
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)}h {int(minutes)}m {int(seconds)}s'


cell_pos       = {}
reward         = {}
start_time     = time.time()  # Record the start time

for geom in env.geom:
    x_pos = []
    y_pos = []
    obs_pressure_list = []  #to store the evolution of the observations 
    reward_list       = []
    # Initialize the environment and get it's state
    env.geometry_selection(geom)
    state, obs_pressure_dic  = env.reset()
    state     = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    obs_pressure_list.append(obs_pressure_dic["pressure"])
    for t in count():
        action = best_net(state) 
        outside_indices = env.check_feasible_actions()
        if outside_indices.size > 0:
            action[:,outside_indices] = 0
        action = action.max(1)[1].view(1, 1)
        observation, reward[geom], terminated, obs_pressure_data,_ = env.step(action.item())
        done = terminated #or truncated
        
        x_pos.append(env.x_c)
        y_pos.append(env.y_c)
        obs_pressure_list.append(obs_pressure_data)
        reward_list.append(reward[geom])
        
        # Move to the next state
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state      = next_state
        
        if done:
            break
      
   
    # Store the position data for this geom in the dictionary
    cell_pos[geom] = {'x': x_pos, 'y': y_pos}
    # Store the pressure in numpy
    obs_pressure   = np.vstack(obs_pressure_list)
    # Store the reward 
    reward_evo     = np.vstack(reward_list)
    
# Calculate elapsed time
elapsed_time = time.time() - start_time

print('----------------')
print('End of simulation')
print('Elapsed time: ' + format_time(elapsed_time))
print('-----------------')    


# Postprocessing
plot_validation_results(cell_pos)