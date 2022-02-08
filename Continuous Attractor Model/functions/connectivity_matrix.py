import numpy as np
from matplotlib import pyplot as plt

def connectivity_matrix(args):
    """
    
    """

    a = args['a']
    gamma = args['gamma']
    beta = args['beta']

    w_0 = lambda position: (a*np.exp(-gamma*np.sum(position**2))) - np.exp(-beta*np.sum(position**2))

    weight_matrix = np.zeros((args['nb_cells_y'], args['nb_cells_x']))
    




    return weight_matrix