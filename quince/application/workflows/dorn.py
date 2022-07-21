import numpy as np
from quince.library.datasets.synthetic import Synthetic
from quince.library.utils import alpha_fn, beta_fn
from quince.library.datasets.utils import e_x_fn
from tqdm import tqdm


def mu_bound(e_x:float, alpha_x: float, beta_x: float, Y:np.ndarray, quantile_y:float, u_up:bool):
    Y_thresh = Y.copy()
    if u_up:
        Y_thresh[Y <= quantile_y] = 0
    else:
        Y_thresh[Y >= quantile_y] = 0
    return (alpha_x * np.mean(Y) + (beta_x - alpha_x) * np.mean(Y_thresh))*e_x


def make_m_experiments(m: int, gamma_star:float, num_examples:int):
    Y0 = []
    Y1 = []
    X = []
    for i in range(m):
        synthetic_dataset = Synthetic(num_examples=num_examples, gamma_star=gamma_star, mode="mu", seed=i)
        X.append(synthetic_dataset.x.flatten())
        Y0.append(synthetic_dataset.y0)
        Y1.append(synthetic_dataset.y1)
    return np.stack(X, axis=0), np.stack(Y0, axis=0), np.stack(Y1, axis=0)


def get_quantile_values(gamma, y:np.ndarray, mirror_tau=False):
    if mirror_tau:
        quantile = 1 - gamma/(gamma + 1)
    else:
        quantile = gamma/(gamma + 1)
    res = np.quantile(a=y.flatten(), q=quantile)
    return res


def dorn_bounds(e_x, alpha_x, beta_x, Y0: np.ndarray, Y1: np.ndarray, Q_E_T_plus, Q_E_C_minus, Q_E_T_minus, Q_E_C_plus):

    y0 = Y0.flatten()
    y1 = Y1.flatten()
    mu_T_sup = mu_bound(e_x, alpha_x, beta_x, y1, quantile_y=Q_E_T_plus, u_up=True)
    mu_T_inf = mu_bound(e_x, alpha_x, beta_x, y1, quantile_y=Q_E_T_minus, u_up=False)
    mu_C_sup = mu_bound(e_x, alpha_x, beta_x, y0, quantile_y=Q_E_C_plus, u_up=True)
    mu_C_inf = mu_bound(e_x, alpha_x, beta_x, y0, quantile_y=Q_E_C_minus, u_up=False)
    return mu_T_sup, mu_T_inf, mu_C_sup, mu_C_inf


def calc_bounds(X: np.ndarray, Y0, Y1, x_resolution=0.1):
    """
    :param self:
    :param X: matrix (mxn) x samples for all experimnets
    :param Y0: matrix (mxn) every line is new sampling experiment evey column is sample
    :param Y1: matrix (mxn) every line is new sampling experiment evey column is sample
    :return:
    """
    axis_x = []
    mu_T_real = []
    mu_C_real = []
    mu_T_sup = []
    mu_T_inf = []
    mu_C_sup = []
    mu_C_inf = []
    # create qr for both Y0 and Y1
    for i in tqdm(np.arange(-2, 2, x_resolution), desc="Calculating bounds"):
        interval = (i, i+x_resolution)

        x_0 = np.mean(interval)
        axis_x.append(np.mean(interval))

        delta_x_indices = (X >= interval[0]) & (X <= interval[1])
        y0_x = Y0[delta_x_indices]
        y1_x = Y1[delta_x_indices]


        mu_T_real.append(np.mean(y1_x))
        mu_C_real.append(np.mean(y0_x))

        Q_E_T_plus = get_quantile_values(gamma=gamma_star, y=y1_x, mirror_tau=False)
        Q_E_C_minus = get_quantile_values(gamma=gamma_star, y=y0_x, mirror_tau=True)
        Q_E_T_minus = get_quantile_values(gamma=gamma_star, y=y1_x, mirror_tau=True)
        Q_E_C_plus = get_quantile_values(gamma=gamma_star, y=y0_x, mirror_tau=False)

        e_x = e_x_fn(x_0, beta=0.75)
        alpha_x = alpha_fn(e_x, gamma_star)
        beta_x = beta_fn(e_x, gamma_star)

        sup_T, inf_T, sup_C, inf_C = dorn_bounds(e_x=e_x,
                                                 alpha_x=alpha_x,
                                                 beta_x=beta_x,
                                                 Y0=y0_x,
                                                 Y1=y1_x,
                                                 Q_E_T_plus=Q_E_T_plus,
                                                 Q_E_C_minus=Q_E_C_minus,
                                                 Q_E_T_minus=Q_E_T_minus,
                                                 Q_E_C_plus=Q_E_C_plus)
        mu_T_sup.append(sup_T)
        mu_T_inf.append(inf_T)
        mu_C_sup.append(sup_C)
        mu_C_inf.append(inf_C)

    return {
        "mu_T_real": np.stack(mu_T_real, axis=0),
        "mu_C_real": np.stack(mu_C_real, axis=0),
        "mu_T_sup": np.stack(mu_T_sup, axis=0),
        "mu_T_inf": np.stack(mu_T_inf, axis=0),
        "mu_C_sup": np.stack(mu_C_sup, axis=0),
        "mu_C_inf": np.stack(mu_C_inf, axis=0),
        "axis_x": np.array(axis_x)
    }


seed = 1
num_examples = 5000
gamma_star = 3.1
m = 10
X, Y0, Y1 = make_m_experiments(m, gamma_star,num_examples)

res = calc_bounds(X, Y0, Y1, x_resolution=0.05)

axis_x = res["axis_x"]
mu_T_real = res["mu_T_real"]
mu_C_real = res["mu_C_real"]
mu_T_sup = res["mu_T_sup"]
mu_T_inf = res["mu_T_inf"]
mu_C_sup = res["mu_C_sup"]
mu_C_inf = res["mu_C_inf"]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(axis_x, mu_T_real, color='blue', label="real mu_T")
ax.plot(axis_x, mu_C_real, color='green', label="real mu_C")

ax.plot(axis_x, mu_T_sup, color='black', linestyle = 'dashed', label="mu_T bounds")
ax.plot(axis_x, mu_T_inf, color='black', linestyle = 'dashed')


ax.plot(axis_x, mu_C_sup, color='red', linestyle = 'dashed', label="mu_C bounds")
ax.plot(axis_x, mu_C_inf, color='red', linestyle = 'dashed')

ax.grid()

ax.legend()
fig.show()

fig, ax = plt.subplots()
ax.plot(axis_x, mu_T_real - mu_C_real, color='blue', label="real tau")
ax.plot(axis_x, mu_T_sup - mu_C_inf, color='black', linestyle = 'dashed', label="tau bounds")
ax.plot(axis_x, mu_T_inf - mu_C_sup, color='black', linestyle = 'dashed')
ax.grid()
ax.legend()
fig.show()
