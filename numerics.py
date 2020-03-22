from typing import Callable
import numpy as np


def gquad332d(integrand: Callable, amin, amax, bmin, bmax):


    # Gauss quadrature. 3x3 points, 2 dimensional
    ahalf = (amax-amin)/2
    bhalf = (bmax-bmin)/2

    amid = (amin + amax)/2
    bmid = (bmin + bmax)/2

    qdist = np.sqrt(3/5)
    result = integrand(amin,bmin)
    result = np.zeros(result.shape)

    weights = [25/81, 40/81, 25/81, 40/81, 64/81, 40/81, 25/81, 40/81, 25/81]
    for i in range(3):
        for j in range(3):
            apt = amid - qdist*ahalf + i*qdist
            bpt = bmid - qdist*bhalf + j*qdist

            result = np.add(result, integrand(apt,bpt)*weights[3*i+j])

    return result

def gquad331d(integrand: Callable, amin, amax, debug=False):
    # Gauss quadrature. 3x3 points. 1 dimensional
    ahalf = (amax-amin)/2

    amid = (amin + amax)/2

    dists = [
        amid - np.sqrt(3/5),
        amid,
        amid + np.sqrt(3/5)
    ]
    weights = [
        5/9,
        8/9,
        5/9
    ]

    # cheesy, expensive way to get the right shape
    result = integrand(amin)
    result = np.zeros(result.shape)

    for dist,weight in zip(dists,weights):
        result = np.add(result, integrand(dist)*weight)

    return result
