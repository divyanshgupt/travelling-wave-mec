
from brian2 import *
from src.params import *

def set_synapses(exc_populations, all_populations):
    """

    """
    exc_to_all_model = """
                        w : 1
                        """
    inh_to_exc_model = '''
                        w : 1 
                        '''
    S = []
    index = 0
    P_i = all_populations[-1]
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

    return S