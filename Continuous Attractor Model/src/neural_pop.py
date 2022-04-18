import src
from src.params import *
from brian2 import *

# Neural Populations
def generate_populations(N):
    
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

    State_i = StateMonitor(P_i, 'v', record=True)
    State_n = StateMonitor(P_n, 'v', record=True)
    State_e = StateMonitor(P_e, 'v', record = True)
    State_w = StateMonitor(P_w, 'v', record = True)
    State_s = StateMonitor(P_s, 'v', record = True)

    neural_pops = [P_n, P_e, P_w, P_s, P_i]
    spike_mons = [M_n, M_e, M_w, M_s, M_i]
    state_mons = [State_n, State_e, State_w, State_s, State_i]
 
    return neural_pops, spike_mons, state_mons

    