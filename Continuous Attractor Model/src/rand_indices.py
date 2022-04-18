import numpy as np

def rand_indices(num_idxs, total_num):
    """
    """

    idxs = np.random.randint(0, total_num, num_idxs)
    idxs = np.sort(idxs)

    return idxs



def rec_indices():
    """
    
    """