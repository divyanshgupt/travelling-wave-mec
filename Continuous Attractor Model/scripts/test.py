from brian2 import *
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation



start_scope() # creat a new scope


# Parameters
n = 80
N = 232 * 232 # Neurons per population
N = n * n

tau_m_plus = 40*ms # Exc. membrane time constant
tau_m_minus = 20*ms # Inh. membrane time constant
tau_s_plus_plus = 5*ms # Exc.-to-exc. synaptic delay
tau_s_minus_plus = 2*ms # Exc.-to-inh. synaptic delay
tau_s_minus = 2*ms # Inh. synaptic delay
a_max_plus = 2 # Exc. drive maximum
a_min_plus = 0.8 # Exc. drive minimum
rho_a_plus = 1.2 * (n/232) # Exc. drive scaled speed
# a_mag_minus = 0.72 # Inh. drive magnitude
a_th_minus = 0.2 # Inh. drive theta amplitude
f = 8*hertz # Inh. drive theta frequency

a_mag_minus = 0.9 # Inh. drive magnitude
w_mag_plus = 0.2  # Exc. synaptic strength
r_w_plus = 6  # Exc. synaptic spread
w_mag_minus = 2.8 # Inh. synaptic strength
r_w_minus = 12 # Inh. synaptic distance
exc_xi = 3 # Exc. synaptic shift

alpha = 0.25*second/metre # Exc. velocity gain
sig_zeta_P = 0.002 # Exc. noise std. dev
sig_zeta_I = 0.002 # Inh. noise std. dev
duration = 1000*ms

defaultclock.dt = 0.1*ms

eqns_exc_n = '''

x = i % sqrt(N) : 1
y = i // sqrt(N): 1

# Specify preferred direction
dir_x = 0 : 1
dir_y = 1 : 1

# Distance from centre
rho = rho_value(x, y, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + sig_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*((dir_x * V_x(t)) + (dir_y * V_y(t))))/tau_m_plus : 1

'''

eqns_exc_s = '''

x = i % sqrt(N) : 1
y = i // sqrt(N): 1

# Specify preferred direction
dir_x = 0 : 1
dir_y = -1 : 1

# Distance from centre
rho = rho_value(x, y, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + sig_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*((dir_x * V_x(t)) + (dir_y * V_y(t))))/tau_m_plus : 1
'''

eqns_exc_e = '''

x = i % sqrt(N) : 1
y = i // sqrt(N): 1

# Specify preferred direction
dir_x = 1 : 1
dir_y = 0 : 1

# Distance from centre
rho = rho_value(x, y, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + sig_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*((dir_x * V_x(t)) + (dir_y * V_y(t))))/tau_m_plus : 1
'''

eqns_exc_w = '''

x = i % sqrt(N) : 1
y = i // sqrt(N): 1

# Specify preferred direction
dir_x = -1 : 1
dir_y = 0 : 1

# Distance from centre
rho = rho_value(x, y, N) : metre (constant over dt)

a_plus = a_plus_value(rho / metre) : 1 (constant over dt)

dv/dt = -v/tau_m_plus  + sig_zeta_P*xi*tau_m_plus**-0.5 + a_plus*(1 + alpha*((dir_x * V_x(t)) + (dir_y * V_y(t))))/tau_m_plus : 1
'''


""" 
eqns_exc = '''

dv/dt = -v/tau_m_plus  + sig_zeta_P*xi*tau_m_plus**-0.5 + a_plus/tau_m_plus : 1

'''  """

eqns_inh = '''

x = i % sqrt(N) : 1
y = i // sqrt(N): 1

dv/dt = -(v - a_minus)/tau_m_minus + sig_zeta_I*xi*tau_m_minus**-0.5 : 1

a_minus = a_mag_minus - a_th_minus*cos(2*pi*f*t): 1

'''

reset = '''
v = 0
'''
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

def zero_velocity(dt, duration):
    nb_steps = int(duration/dt)
    x = zeros(nb_steps)
    y = zeros(nb_steps)
    velocity = column_stack((x, y))
    return velocity

velocity = zero_velocity(defaultclock.dt, duration)

dt = defaultclock.dt
V_x = TimedArray(velocity[:, 0]*metre/second, dt=dt)
V_y = TimedArray(velocity[:, 1]*metre/second, dt=dt)


print("Running the simulation")
run(duration)
print("Simulation finished")