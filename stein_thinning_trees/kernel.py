"""Kernel functions."""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.spatial.distance import squareform
from stein_thinning.util import isfloat

# Placeholder for the correct implementation of kc_distance
def kc_distance(tree1, tree2, lambda_=0.0):
    return tree1.kc_distance(tree2, lambda_=lambda_)

def vfk0_imq(a, b, sa, sb, linv):
    amb = a.T - b.T
    qf = 1 + np.sum(np.dot(linv, amb) * amb, axis=0)
    t1 = -3 * np.sum(np.dot(np.dot(linv, linv), amb) * amb, axis=0) / (qf ** 2.5)
    t2 = (np.trace(linv) + np.sum(np.dot(linv, sa.T - sb.T) * amb, axis=0)) / (qf ** 1.5)
    t3 = np.sum(sa.T * sb.T, axis=0) / (qf ** 0.5)
    return t1 + t2 + t3

def make_precon(trees, scr, pre='id'):
    # Sample size
    sz = len(trees)

    # Squared pairwise median using KC distance
    def med2(m):
        if sz > m:
            sub = trees[np.linspace(0, sz - 1, m, dtype=int)]
        else:
            sub = trees
        dists = [kc_distance(sub[i], sub[j]) for i in range(len(sub)) for j in range(i + 1, len(sub))]
        return np.median(dists) ** 2

    # Select preconditioner
    m = 1000
    if pre == 'id':
        linv = np.identity(sz)
    elif pre == 'med':
        m2 = med2(m)
        if m2 == 0:
            raise Exception('Too few unique samples in trees.')
        linv = inv(m2 * np.identity(sz))
    elif pre == 'sclmed':
        m2 = med2(m)
        if m2 == 0:
            raise Exception('Too few unique samples in trees.')
        linv = inv(m2 / np.log(np.minimum(m, sz)) * np.identity(sz))
    elif pre == 'smpcov':
        c = np.cov([kc_distance(tree, tree) for tree in trees], rowvar=False)
        if not all(eig(c)[0] > 0):
            raise Exception('Too few unique samples in trees.')
        linv = inv(c)
    elif isfloat(pre):
        linv = inv(float(pre) * np.identity(sz))
    else:
        raise ValueError('Incorrect preconditioner type.')
    return linv

def make_imq(trees, scr, pre='id'):
    linv = make_precon(trees, scr, pre)
    def vfk0(a, b, sa, sb):
        return vfk0_imq(a, b, sa, sb, linv)
    return vfk0
