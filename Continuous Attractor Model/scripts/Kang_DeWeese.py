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

date_stamp = str(datetime.datetime.today())[:13]
location = src.set_location(f'../data/{date_stamp}')
start_scope() # creat a new scope

dt = defaultclock.dt = 0.1*ms

# # Initialise indices to record from:
# rec_idxs = src.rand_indices(150, )

@implementation('numpy', discard_units=True)
@check_units(x = 1, y = 1, N = 1, result = metre)
def rho_value(x, y, N):

    value = sqrt(((x - ((N+1)/2))**2 + (y - ((N+1)/2))**2)/(N/2))

    return value * metre


@implementation('numpy', discard_units=True)
@check_units(rho = 1, result = 1)
def a_plus_value(rho):

    if rho < rho_a_plus:
        value = (a_max_plus - a_min_plus) * (1 - cos(pi*rho/rho_a_plus))
    else:
        value = a_min_plus
    
    return value
# Generate Neural Populations
neural_pops, spike_mons = src.generate_populations(N)
P_n, P_s, P_e, P_w, P_i = neural_pops
M_n, M_s, M_e, M_w, M_i = spike_mons
# State_i, State_n, State_e, State_w, State_s = state_mons

# Set up synaptic connections
exc_populations = [P_n, P_e, P_w, P_s]
all_populations = [P_n, P_e, P_w, P_s, P_i]

S = src.set_synapses(exc_populations, all_populations)

# Generate velocity array
# trajectory, velocity_array, angle = src.smooth_random_trajectory(n, 0.4, 0.1, 1000)
speed = 0.1 # m/sec
trajectory, velocity_array = src.straight_trajectory(dt, duration, speed)
V_x = TimedArray(velocity_array[:, 0], dt=dt)
V_y = TimedArray(velocity_array[:, 1], dt=dt)

print("Running the simulation")
net = Network(collect())
net.add(spike_mons)


run(duration)

print("Simulation over")

print("Storing the recordings")

spike_rec = (M_n.get_states(['t', 'i']), M_e.get_states(['t', 'i']), M_w.get_states(['t', 'i']), M_s.get_states(['t', 'i']), M_i.get_states(['t', 'i']))
# state_rec = (State_n.get_states('v'), State_e.get_states('v'), State_w.get_states('v'), State_s.get_states('v'), State_i.get_states('v'))


# recordings = (trajectory, velocity_array, state_rec, spike_rec)

recordings = (trajectory, velocity_array, spike_rec)

src.save_data(recordings, location, 'recordings', method='pickle')

print("Task Finished!")