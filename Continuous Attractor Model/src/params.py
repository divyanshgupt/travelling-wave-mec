from brian2 import *

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

""" w_mag_plus = 0.2 * (N/(232*232)) # Exc. synaptic strength
r_w_plus = 6 * (n/232) # Exc. synaptic spread
w_mag_minus = 2.8 * (N/(232*232)) # Inh. synaptic strength
r_w_minus = (12/232) * n # Inh. synaptic distance
exc_xi = (3/232) * n # Exc. synaptic shift """

w_mag_plus = 0.2  # Exc. synaptic strength
r_w_plus = 6  # Exc. synaptic spread
w_mag_minus = 2.8 # Inh. synaptic strength
r_w_minus = 12 # Inh. synaptic distance
exc_xi = 3 # Exc. synaptic shift


alpha = 0.25*second/metre # Exc. velocity gain
# var_zeta_P = 0.002**2 # Exc. noise magnitude
# var_zeta_I = 0.002**2 # Inh. noise magnitude

sig_zeta_P = 0.002 # Exc. noise std. dev
sig_zeta_I = 0.002 # Inh. noise std. dev

duration = 1000*ms
