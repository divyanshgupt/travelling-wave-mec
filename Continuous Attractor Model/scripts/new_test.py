from brian2 import *
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import src
from src.params import *


print("Initializing rat trajectory")
print(f'Straight Trajectory Function type:{type(src.straight_trajectory)}')
dt = defaultclock.dt
trajectory, velocity = src.straight_trajectory(dt, duration, 0.1)

start_scope() # creat a new scope

date_stamp = str(datetime.datetime.today())[:13]
location = src.set_location(f'../data/{date_stamp}')

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

V_x = TimedArray(velocity[:, 0], dt=dt)
V_y = TimedArray(velocity[:, 1], dt=dt)
print("Trajectory set!")

# Neural Populations

## North
P_n = NeuronGroup(N, src.eqns_exc_n, threshold='v > 1', reset=src.reset, method='euler')
P_n.v = 'rand()'

## South
P_s = NeuronGroup(N, src.eqns_exc_s, threshold='v > 1', reset=src.reset, method='euler')
P_s.v = 'rand()'

## East
P_e = NeuronGroup(N, src.eqns_exc_e, threshold='v > 1', reset=src.reset, method='euler')
P_e.v = 'rand()'

## West
P_w = NeuronGroup(N, src.eqns_exc_w, threshold='v > 1', reset=src.reset, method='euler' )
P_w.v = 'rand()'

## Inhibitory
P_i = NeuronGroup(N, src.eqns_inh, threshold='v > 1', reset=src.reset, method='euler' )
P_i.v = 'rand()'

M_n = SpikeMonitor(P_n)
M_s = SpikeMonitor(P_s)
M_e = SpikeMonitor(P_e)
M_w = SpikeMonitor(P_w)
M_i = SpikeMonitor(P_i)

# State_i = StateMonitor(P_i, 'v', record=True)
# State_n = StateMonitor(P_n, 'v', record=True)
# State_e = StateMonitor(P_e, 'v', record=True)
# State_w = StateMonitor(P_w, 'v', record=True)
# State_s = StateMonitor(P_s, 'v', record=True)

S = []
exc_populations = [P_n, P_e, P_w, P_s]
all_populations = [P_n, P_e, P_w, P_s, P_i]
index = 0

exc_to_all_model = """
                    w : 1
                    """
inh_to_exc_model = '''
                    w : 1 
                    '''


print("Setting exc >> all connections")
for src in exc_populations:
    for trg in all_populations:
        print("Synapse group index:", index)
        S.append(Synapses(src, trg, exc_to_all_model, on_pre='v_post += w'))
        # dir_x = src.dir_x
        # dir_y = src.dir_y
        if src == trg:
            S[index].connect(condition='sqrt((x_post - x_pre -exc_xi*dir_x_pre)**2 + (y_post - y_pre - exc_xi*dir_y_pre)**2) < r_w_plus and i!=j')
        else:
            S[index].connect(condition='sqrt((x_post - x_pre -exc_xi*dir_x_pre)**2 + (y_post - y_pre - exc_xi*dir_y_pre)**2) < r_w_plus')

        S[index].w = 'w_mag_plus*((1 + cos(pi*sqrt((x_post - x_pre -exc_xi*dir_x_pre)**2 + (y_post - y_pre - exc_xi*dir_y_pre)**2)/r_w_plus))/2)'

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
    S[index].connect(condition = 'sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2) < 2 * r_w_minus')
    S[index].delay = tau_s_minus
    S[index].w = "- w_mag_minus*((1 - cos(pi*sqrt((x_post - x_pre)**2 + (y_post - y_pre)**2)/r_w_plus))/2)"
    index += 1

print("Connections set!")

# def straight_trajectory(dt, duration, speed):
#     """
    
#     Args:
#         dt - 
#         duration - 
#         speed - in metres/sec
#     """
    
#     nb_steps = int(duration/dt)
#     angle = np.random.random()*2*pi
    
#     x = cos(angle)*arange(0, nb_steps+1)*speed*dt
#     y = sin(angle)*arange(0, nb_steps+1)*speed*dt

#     velocity_x = diff(x)/dt
#     velocity_y = diff(y)/dt

#     velocity = column_stack((velocity_x, velocity_y))
#     trajectory = column_stack((x, y))

#     return trajectory, velocity



print("Running the simulation")
run(duration)
print("Simulation finished")

print("Storing data")
spike_rec = (M_n.get_states(['t', 'i']), M_e.get_states(['t', 'i']), M_w.get_states(['t', 'i']), M_s.get_states(['t', 'i']), M_i.get_states(['t', 'i']))
# state_rec = (State_n.get_states('v'), State_e.get_states('v'), State_w.get_states('v'), State_s.get_states('v'), State_i.get_states('v'))

# recordings = (trajectory, velocity_array, state_rec, spike_rec)
recordings = (trajectory, velocity, spike_rec)
src.save_data(recordings, location, 'recordings', method='pickle')
print("Task Finished!")