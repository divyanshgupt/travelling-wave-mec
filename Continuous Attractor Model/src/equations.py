from brian2 import *
from src.params import *

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

# @implementation('numpy', discard_units=True)
# @check_units(x = 1, y = 1, N = 1, result = metre)
# def rho_value(x, y, N):

#     value = sqrt(((x - ((N+1)/2))**2 + (y - ((N+1)/2))**2)/(N/2))

#     return value * metre


# @implementation('numpy', discard_units=True)
# @check_units(rho = 1, result = 1)
# def a_plus_value(rho):

#     if rho < rho_a_plus:
#         value = (a_max_plus - a_min_plus) * (1 - cos(pi*rho/rho_a_plus))
#     else:
#         value = a_min_plus
    
#     return value
