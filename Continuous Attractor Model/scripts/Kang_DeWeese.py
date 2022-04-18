#!/usr/bin/env python
# coding: utf-8

from brian2 import *
#import numpy as np
#from matplotlib import pyplot as plt
import pickle
import datetime
import os
import src
from src.params import *

data_folder = 'data/' + str(datetime.datetime.today())[:16]
location = os.path.abspath(data_folder)
location = os.path.join(os.getcwd(), location)
os.makedirs(location)

start_scope() # creat a new scope

dt = defaultclock.dt = 0.1*ms

# Initialise indices to record from:
rec_idxs = src.rand_indices(150, )

# Generate Neural Populations
neural_pops, spike_mons, state_mons = src.generate_populations(N, rec_idxs)
P_n, P_s, P_e, P_w, P_i = neural_pops
M_n, M_s, M_e, M_w, M_i = spike_mons
State_i, State_n, State_e, State_w, State_s = state_mons

# Set up synaptic connections
exc_populations = [P_n, P_e, P_w, P_s]
all_populations = [P_n, P_e, P_w, P_s, P_i]

S = src.set_synapses(exc_populations, all_populations)

# Generate velocity array
trajectory, velocity_array, angle = src.smooth_random_trajectory(n, 0.4, 0.1, 1000)
V_x = TimedArray(velocity_array[:, 0], dt=dt)
V_y = TimedArray(velocity_array[:, 1], dt=dt)

print("Running the simulation")
run(duration)

print("Simulation over")

# S:
# * 0 - 4 : north > north, south, east, west, inh
# * 5 - 9 : south > north, south, east, west, inh
# * 10 - 14: east > north, south, east, west, inh

print("Storing the recordings")

spike_rec = (M_n.get_states(['t', 'i']), M_e.get_states(['t', 'i']), M_w.get_states(['t', 'i']), M_s.get_states(['t', 'i']), M_i.get_states(['t', 'i']))
state_rec = (State_n.get_states('v'), State_e.get_states('v'), State_w.get_states('v'), State_s.get_states('v'), State_i.get_states('v'))


recordings = (trajectory, velocity_array, state_rec, spike_rec)

recordings_filename = location + '/recordings'
recordings_file = open(recordings_filename, 'wb')
pickle.dump(recordings, recordings_file)
recordings_file.close()

print("Task Finished!")