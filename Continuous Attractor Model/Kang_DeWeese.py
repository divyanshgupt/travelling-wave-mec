#!/usr/bin/env python
# coding: utf-8


from brian2 import *
#import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import datetime
import os


data_folder = 'data/' + str(datetime.datetime.today())[:16]
location = os.path.abspath(data_folder)
location = os.path.join(os.getcwd(), location)
os.makedirs(location)

start_scope() # creat a new scope


# Parameters
n = 40
N = 232 * 232 # Neurons per population
N = n * n

tau_m_plus = 40*ms # Exc. membrane time constant
tau_m_minus = 20*ms # Inh. membrane time constant
tau_s_plus_plus = 5*ms # Exc.-to-exc. synaptic delay
tau_s_minus_plus = 2*ms # Exc.-to-inh. synaptic delay
tau_s_minus = 2*ms # Inh. synaptic delay
a_max_plus = 2 # Exc. drive maximum
a_min_plus = 0.8 # Exc. drive minimum
rho_a_plus = 1.2 # Exc. drive scaled speed
a_mag_minus = 0.72 # Inh. drive magnitude
a_th_minus = 0.2 # Inh. drive theta amplitude
f = 8*hertz # Inh. drive theta frequency
w_mag_plus = 0.2 # Exc. synaptic strength
r_w_plus = (6/232) * n # Exc. synaptic spread
w_mag_minus = 2.8 # Inh. synaptic strength
r_w_minus = (12/232) * n # Inh. synaptic distance
xi = (3/232) * n # Exc. synaptic shift
alpha = 0.25*second/metre # Exc. velocity gain
var_zeta_P = 0.002**2 # Exc. noise magnitude
var_zeta_I = 0.002**2 # Inh. noise magnitude


duration = 1000*ms

defaultclock.dt = 0.1*ms



@implementation('numpy', discard_units=True)
@check_units(dir_x = metre, dir_y = metre, V_x = metre/second, V_y = metre/second, result= metre/second)
def dot_product(dir_x, dir_y, V_x, V_y):
    dir = [dir_x, dir_y]
    V = [V_x, V_y]
    product = dot(dir, V)
    print(product)
    return product * metre / second



dot_product(3 * metre, 2*metre, 1*metre/second, 5*metre/second)




eqns_exc_n = '''

x = i % sqrt(N) * metre : metre 
y = i // sqrt(N) * metre : metre

# Specify preferred direction
dir_x = 0 * metre : metre
dir_y = 1 * metre: metre

# Distance from centre
rho = rho_value(x / metre, y / metre, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + var_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*dot_product(dir_x, dir_y, V_x(t), V_y(t)))/tau_m_plus : 1

'''

eqns_exc_s = '''

x = i % sqrt(N) * metre : metre
y = i // sqrt(N) * metre : metre

# Specify preferred direction
dir_x = 0 * metre : metre
dir_y = -1 * metre : metre

# Distance from centre
rho = rho_value(x / metre, y / metre, N) : metre (constant over dt)
 
a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + var_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*dot_product(dir_x, dir_y, V_x(t), V_y(t)))/tau_m_plus : 1
'''

eqns_exc_e = '''

x = i % sqrt(N) * metre : metre
y = i // sqrt(N) * metre : metre

# Specify preferred direction
dir_x = 1 * metre : metre
dir_y = 0 * metre : metre

# Distance from centre
rho = rho_value(x / metre, y / metre, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + var_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*dot_product(dir_x, dir_y, V_x(t), V_y(t)))/tau_m_plus : 1
'''

eqns_exc_w = '''

x = i % sqrt(N) * metre : metre
y = i // sqrt(N) * metre : metre

# Specify preferred direction
dir_x = -1 * metre : metre
dir_y = 0 * metre : metre

# Distance from centre
rho = rho_value(x / metre, y / metre, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + var_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*dot_product(dir_x, dir_y, V_x(t), V_y(t)))/tau_m_plus : 1
'''


""" 
eqns_exc = '''

dv/dt = -v/tau_m_plus  + var_zeta_P*xi*tau_m_plus**-0.5 + a_plus/tau_m_plus : 1

'''  """

eqns_inh = '''

x = i % sqrt(N) * metre : metre
y = i // sqrt(N) * metre : metre

dv/dt = -(v - a_minus)/tau_m_minus + var_zeta_I*xi*tau_m_minus**-0.5 : 1
a_minus = a_mag_minus - a_th_minus*cos(2*pi*f*t): 1

'''

reset = '''
v = 0
'''



#@check_units(i = 1, result = [metre for x in range(len(i))])
@check_units(i = 1, N = 1, result = metre)
def location_x(i, N):
    x = i % N
    return x * metre

@check_units(i = 1, N = 1, result = metre)
def location_y(i, N):
    y = i // N
    return y * metre

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

@check_units(x = 1, y = 1, result = metre)
def dir_array(x, y, N):
    x_array = x * ones(N)
    y_array = y * ones(N)

    return column_stack((x_array, y_array)) * metre


# Neural Populations

## North
P_n = NeuronGroup(N, eqns_exc_n, threshold='v > 1', reset=reset, method='euler')
P_n.v = 'rand()'

## South
P_s = NeuronGroup(N, eqns_exc_s, threshold='v > 1', reset=reset, method='euler')
P_s.v = 'rand()'

## East
P_e = NeuronGroup(N, eqns_exc_e, threshold='v > 1', reset=reset, method='euler')
P_e.v = 'rand()'

## West
P_w = NeuronGroup(N, eqns_exc_w, threshold='v > 1', reset=reset, method='euler' )
P_w.v = 'rand()'

## Inhibitory
P_i = NeuronGroup(N, eqns_inh, threshold='v > 1', reset=reset, method='euler' )
P_i.v = 'rand()'

M_n = SpikeMonitor(P_n)
M_s = SpikeMonitor(P_s)
M_e = SpikeMonitor(P_e)
M_w = SpikeMonitor(P_w)
M_i = SpikeMonitor(P_i)

State_i = StateMonitor(P_i, 'v', record=True)
State_n = StateMonitor(P_n, 'v', record=True)
State_e = StateMonitor(P_e, 'v', record = True)
State_w = StateMonitor(P_w, 'v', record = True)
State_s = StateMonitor(P_s, 'v', record = True)




S = []
exc_populations = [P_n, P_e, P_w, P_s]
all_populations = [P_n, P_e, P_w, P_s, P_i]
index = 0

exc_to_all_model = """
                    r = sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2) : 1 (constant over dt)
                    w : 1
                    """
inh_to_exc_model = '''
                    r = sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2) : 1 (constant over dt)
                    w : 1 
                    '''


print("Setting exc >> all connections")
for src in exc_populations:
    for trg in all_populations:
        print("Synapse group index:", index)
        S.append(Synapses(src, trg, exc_to_all_model, on_pre='v_post += w'))
        if src == trg:
            S[index].connect(condition='sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2)/metre < r_w_plus and i!=j')
        else:
            S[index].connect(condition='sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2)/metre < r_w_plus')
        S[index].w = 'w_mag_plus**((1 + cos(pi*r/r_w_plus))/2)'

        # Synaptic Delay
        if trg != P_i:
            S[index].delay = tau_s_plus_plus
        else:
            S[index].delay = tau_s_minus_plus
        
        index += 1

# Inhibitory to excitatory connections
print("Setting inh >> exc connections")
for trg in exc_populations:
    print("Synapse group index:", index)
    S.append(Synapses(P_i, trg, inh_to_exc_model, on_pre = 'v_post += w'))
    S[index].connect(condition = 'sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2)/metre < 2 * r_w_minus')
    S[index].delay = tau_s_minus
    S[index].w = "- w_mag_minus**((1 + cos(pi*r/r_w_plus))/2)"
    index += 1


#@title Simulated Velocity Inputs:

def simulate_random_velocity(duration, dt, step_size):
    """
    Inputs
    
    """
    x = step_size * cumsum((random(int(duration/dt)) - 0.5))
    y = step_size * cumsum((random(int(duration/dt)) - 0.5))

    # velocity = column_stack((x, y))

    return x / second, y / second

step_size = 0.001*metre
dt = 0.1*ms
velocity_array_x, velocity_array_y = simulate_random_velocity(duration, dt, step_size)


fig, ax = subplots()
ax.plot(velocity_array_x[:], velocity_array_y[:])
ax.plot(velocity_array_x[0], velocity_array_y[0], 'ro', color='black', label='start')
ax.plot(velocity_array_x[-1], velocity_array_y[-1], 'ro', color='blue', label='stop')
fig.suptitle("Rat Trajectory")
ax.legend()
fig.savefig(location + '/animal_velocity.png')
close(fig)

V_x = TimedArray(velocity_array_x, dt=dt)
V_y = TimedArray(velocity_array_y, dt=dt)


# print(V_x(1*second))



print("Running the simulation")
run(duration)

fig, ax = subplots()
for i in range(N):
    ax.plot(State_n.v[i, :])
fig.suptitle("Membrane potential for neurons in P_n over time")
ax.set_ylabel("Membrane Potential")
ax.set_xlabel("Time")
fig.savefig(location + '/animal_velocity.png')
close(fig)
# ## Plot Connectivity

# S:
# * 0 - 4 : north > north, south, east, west, inh
# * 5 - 9 : south > north, south, east, west, inh
# * 10 - 14: east > north, south, east, west, inh
# 

recordings = (State_n, State_e, State_w, State_s, State_i, M_n, M_e, M_w, M_s, M_i)


recordings_filename = location + '/recordings'
recordings_file = open(recordings_filename, 'wb')
pickle.dump(recordings, recordings_file)
recordings_file.close()