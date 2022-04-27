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

print(f"Number of neurons per population: {N}")
date_stamp = str(datetime.datetime.today())[:13]
location = src.set_location(f'data/{date_stamp}')
start_scope() # creat a new scope

dt = defaultclock.dt = 0.1*ms

# # Initialise indices to record from:
# rec_idxs = src.rand_indices(150, )

@implementation('numpy', discard_units=True)
@check_units(x = 1, y = 1, n = 1, result = metre)
def rho_value(x, y, n):

    value = sqrt(((x - ((n+1)/2))**2 + (y - ((n+1)/2))**2))/(n/2)

    return value * metre


@implementation('numpy', discard_units=True)
@check_units(rho = 1, result = 1)
def a_plus_value(rho):

    if rho < rho_a_plus:
        value = a_min_plus + (a_max_plus - a_min_plus) * ((1 + cos(pi*rho/rho_a_plus))/2)
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

print("Connections set")

# Generate velocity array
def straight_trajectory(dt, duration, speed):
    """
    Args:
        dt - 
        duration - 
        speed - in metres/sec
    """
    nb_steps = int(duration/dt)
    angle = np.random.random()*2*pi   
    x = cos(angle)*arange(0, nb_steps+1)*speed*dt
    y = sin(angle)*arange(0, nb_steps+1)*speed*dt
    velocity_x = diff(x)/dt
    velocity_y = diff(y)/dt

    velocity = column_stack((velocity_x, velocity_y))
    trajectory = column_stack((x, y))

    return trajectory, velocity

def zero_velocity(dt, duration):
    nb_steps = int(duration/dt)
    x = zeros(nb_steps)
    y = zeros(nb_steps)
    velocity = column_stack((x, y))
    trajectory = column_stack((x, y))
    return trajectory, velocity


print("Initializing rat trajectory")
# print(f'Straight Trajectory Function type:{type(src.straight_trajectory)}')
dt = defaultclock.dt
trajectory, velocity = zero_velocity(defaultclock.dt, duration)

# trajectory, velocity = straight_trajectory(dt, duration, 0.1)

V_x = TimedArray(velocity[:, 0] * metre/second, dt=dt)
V_y = TimedArray(velocity[:, 1] * metre/second, dt=dt)
print("Trajectory set!")

print("Running the simulation")
net = Network(collect())
net.add(spike_mons)
net.add(S)
net.run(duration)

# run(duration)

print("Simulation over")

print("Storing the recordings")

spike_rec = (M_n.get_states(['t', 'i']), M_e.get_states(['t', 'i']), M_w.get_states(['t', 'i']), M_s.get_states(['t', 'i']), M_i.get_states(['t', 'i']))
# state_rec = (State_n.get_states('v'), State_e.get_states('v'), State_w.get_states('v'), State_s.get_states('v'), State_i.get_states('v'))


# recordings = (trajectory, velocity_array, state_rec, spike_rec)

recordings = (trajectory, velocity, spike_rec)

src.save_data(recordings, location, 'recordings', method='pickle')

print("Task Finished!")