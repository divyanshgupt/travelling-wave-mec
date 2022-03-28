#!/usr/bin/env python
# coding: utf-8



from brian2 import *
#import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm

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





def exc_to_any_connectivity(N, dir_x, dir_y, same_pop=False):
    """
    Sets up the connectivity matrix between two excitatory populations.
    N - number of neurons in each population
    dir - directional tuning vector of pre_synaptic population
    """
    connectivity = zeros((N, N))
    # connectivity = zeros(N**2) 
   # dir = eval(dir) # convert string representation to list
    for i in range(N): # looping over source neurons
        i_x = i % N
        i_y = i // N
        for j in range(N): # looping over target neurons
            j_x = j % N
            j_y = j % N
            
            if same_pop and i ==  j:
                connectivity[i, j] = 0
            else:
                r_x = j_x - i_x - xi*dir_x
                r_y = j_y - i_y - xi*dir_y
                distance = linalg.norm(array([r_x, r_y]))

                if distance < r_w_plus:
                    # connectivity[i, j] = w_mag_plus**(1 + cos(pi*sqrt((j_x - i_x - xi*dir_x)**2 + (j_y - i_y - xi*dir_y)**2)))
                    connectivity[i, j] = w_mag_plus**((1 + cos(pi*distance/r_w_plus))/2)

    return connectivity.flatten()


def inh_to_exc_connectivity(N):
    """
    Sets up the connectivity matrix between two excitatory populations.
    N - number of neurons in each population
    dir - directional tuning vector of pre_synaptic population
    """
    connectivity = empty((N, N))

    for i in range(N): # looping over source neurons
        i_x = i % N
        i_y = i // N
        for j in range(N):
            j_x = j % N
            j_y = j % N

            r_x = j_x - i_x 
            r_y = j_y - i_y 
            distance = linalg.norm(array([r_x, r_y]))

            if distance < r_w_minus:
                connectivity[i, j] = -w_mag_minus**((1 + cos(pi*distance/r_w_minus))/2)

    return connectivity.flatten()



S = [] # to store  the 25 synapse classes
exc_populations = [P_n, P_s, P_e, P_w]
all_populations = [P_n, P_s, P_e, P_w, P_i]
index = 0

# Set connections from excitatory to excitatory populations: (Total 16 iterations)
print("Setting up exc-->exc connections")
for src in exc_populations:
    print("Source po|pulation:", src.name)
    for trg in exc_populations:
        print("Target population:", trg)
        S.append(Synapses(src, trg, 'w: 1', on_pre='v_post += w'))
        if src == trg: # connection within the population     
            S[index].connect(condition='i!=j') # if connection within population, don't connect neurons to themselves
            # connectivity = exc_to_any_connectivity(N, src.dir, same_pop=True)
            # S[index].w = 'connectivity[i, j]'
            S[index].w = delete(exc_to_any_connectivity(N, src.dir_x/metre, src.dir_y/metre, same_pop=True).flatten(), range(0, N*N, N+1), 0) # deletes diagonal entries of connectivity before assigning it to weights
        else:
            S[index].connect() # if connections are between two populations, connect all neurons
            # connectivity = exc_to_any_connectivity(N, src.dir)
            # S[index].w = 'connectivity[i, j]'
            S[index].w = exc_to_any_connectivity(N, src.dir_x/metre, src.dir_y/metre).flatten()
        S[index].delay = 'tau_s_plus_plus'
        index += 1

# Set connections from excitatory to inhibitory population: (Total 4 iterations)
print("Setting up exc-->inh connections")
for i in exc_populations:
    S.append(Synapses(i, P_i, 'w:1', on_pre='v_post += w'))
    S[index].connect()
    # connectivity = exc_to_any_connectivity(N, src.dir)
    # S[index].w = 'connectivity[i, j]'
    S[index].w = exc_to_any_connectivity(N, src.dir_x/metre, src.dir_y/metre).flatten()
    S[index].delay = 'tau_s_minus_plus'
    index += 1    

# Set connections from inhibitory to excitatory neurons: (Total 4 iterations)
print("Setting up inh-->exc connections")
for i in exc_populations:
    S.append(Synapses(P_i, i, 'w:1', on_pre='v_post += w'))
    S[index].connect()
    # connectivity = inh_to_exc_connectivity(N)
    # S[index].w = 'connectivity[i, j]'
    S[index].w = inh_to_exc_connectivity(N).flatten()
    S[index].delay = 'tau_s_minus'
    index += 1

# The inhibitory population doesn't have recurrent connections within itself


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


# figure(dpi=130)
# plot(velocity_array_x[:], velocity_array_y[:])
# plot(velocity_array_x[0], velocity_array_y[0], 'ro', color='black', label='start')
# plot(velocity_array_x[-1], velocity_array_y[-1], 'ro', color='blue', label='stop')
# title("Rat Trajectory")
# legend()


V_x = TimedArray(velocity_array_x, dt=dt)
V_y = TimedArray(velocity_array_y, dt=dt)




# print(V_x(1*second))



print("Running the simulation")
run(duration)


# ## Plot Connectivity

# S:
# * 0 - 4 : north > north, south, east, west, inh
# * 5 - 9 : south > north, south, east, west, inh
# * 10 - 14: east > north, south, east, west, inh
# 






