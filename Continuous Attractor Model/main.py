import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import functions

args = {'a': 1, # regulates the sign of the synaptic weights
        'nb_steps': ,
        'nb_cells_x': ,
        'nb_cells_y': ,
        'lambda': 13, # default value as chosen by Mittal & Narayanan, 2021
        'l': 2, # amount of shift in connectivity along preferred direction
        

        }

args['beta'] = 3 / (args['lambda']**2)
args['gamma'] = 1.1 * args['beta']


# Initialize 3-D array of zeroes for storing firing rates over time
activity = np.zeros((args['nb_cells_y'], args['nb_cells_x'], args['nb_steps']))

# Initialize preferred directions
pref_directions = functions.preferred_directions(args)

# Initialize connectivity matrix 
weights = functions.connectivity_matrix(pref_directions, args)

# Initialize virtual trajectory
trajectory = functions.virtual_trajectory(args)

# Initialize feedforward input
B = functions.feedforward_input(trajectory, args)

for t in range(args['nb_steps']):
    # Runge-Kutta integration of activity
    activity[:, :, t+1] = 