import numpy as np


def get_num_digits(N):
    """Return the number of digits of the given number N.
    
    Parameters
    ----------
    N : int
        The number we want to apply the function to.
    
    Returns
    -------
    int :
        The numbers of digits needed to represent the number N.
    """
    return int(np.log10(N))+1
