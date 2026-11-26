"""Constants and the random split of the 2015 pool used by lendingclub.py. The pre-2015 splits are in splits.py."""
import numpy as np

B1 = 35000.0                 # largest loss, the maximum loan amount
C_HAT = 'c1_hat'             # column of c_hat in the prediction files


def split_pool3(n_pool, seed, fracs=(0.2, 0.4, 0.4)):
    """Random split of the 2015 pool indices into U (new cases, features only), cal and test."""
    perm = np.random.default_rng(seed).permutation(n_pool)
    k1 = int(n_pool * fracs[0])
    k2 = k1 + int(n_pool * fracs[1])
    return perm[:k1], perm[k1:k2], perm[k2:]