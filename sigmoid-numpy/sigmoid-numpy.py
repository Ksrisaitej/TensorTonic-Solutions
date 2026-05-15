import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    z = np.array(x)
    g = 1./(1+np.exp(-z))
    return g