import numpy as np


def nCr(n_, r_):
    return np.math.factorial(n_) / (np.math.factorial(r_) * np.math.factorial(n_ - r_))
