import math

import numpy as np
import scipy.spatial


def l1_norm(x1, x2):
    return np.linalg.norm(
        np.subtract(x1, x2),
        ord=1
    )

def l2_norm2(x1, x2):
    return np.linalg.norm(
        np.subtract(x1, x2),
        ord=2
    )


def l2_norm1(x1):
    return np.linalg.norm(
        x1,
        ord=2
    )

# Something buggy in this code!!!
def mahalanobis(x1, x2):
    return scipy.spatial.distance.mahalanobis(
        x1.ravel(), x2.ravel(), np.cov(x1.ravel(), x2.ravel())
    )


def correlation_coeff(x1, x2):
    x1=x1.ravel()
    x2=x2.ravel()
    numerator = np.mean((x1 - x1.mean()) * (x2 - x2.mean()))
    denominator = x1.std() * x2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

def intersection(x1, x2):
    result = 0
    for x, y in zip(x1, x2):
        result += min(x, y) / max(x, y)
    return result/(10 ** math.ceil(math.log(abs(result))))


def cosine(x1, x2):
    return 1 - scipy.spatial.distance.cosine(x1.ravel(), x2.ravel())

def linf_norm(x1, x2):
    norm = np.linalg.norm(
        np.subtract(x1, x2),
        ord=np.inf
    )
    return norm/(10 ** math.ceil(math.log10(abs(norm))))