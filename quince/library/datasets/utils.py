import numpy as np

from quince.library import utils


def e_x_fn(x, beta):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    return nominal


def complete_propensity(x, u, gamma, beta=0.75):
    nominal = e_x_fn(x, beta)
    alpha_x = utils.alpha_fn(nominal, gamma)
    beta_x = utils.beta_fn(nominal, gamma)
    return (u / alpha_x) + ((1 - u) / beta_x)



def f_mu(x, t, u, theta=4.0):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        - 2 * np.sin((4 * t - 2) * x)
        - (theta * u - 2) * (1 + 0.5 * x)
    )
    return mu


def linear_normalization(x, new_min, new_max):
    return (x - x.min()) * (new_max - new_min) / (x.max() - x.min()) + new_min
