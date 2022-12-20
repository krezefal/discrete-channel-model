import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from consts import q


def inner_integrand(u: np.float64) -> np.float64:
    return np.exp(- u*u / 2)


def integrand(eta: np.float64, gamma: int) -> np.float64:
    q_func = 1 / np.sqrt(2*np.pi) * quad(lambda u: inner_integrand(u), eta + np.sqrt(2 * gamma), np.inf)[0]
    _1st_mult = (1 - q_func) ** (q-1)
    _2nd_mult = 1 / np.sqrt(2*np.pi)
    _3rd_mult = np.exp(-((eta + np.sqrt(2 * gamma) - np.sqrt(2*gamma))**2) / 2)
    return _1st_mult * _2nd_mult * _3rd_mult


def calc_transition_prob(gamma: int) -> np.float64:
    # eta = np.random.normal(0, 1)
    # z = np.sqrt(2 * gamma) + eta
    integrand_res = quad(lambda eta: integrand(eta, gamma), -np.inf, np.inf)[0]
    return 1 - integrand_res


def render_3d(mtx: np.ndarray, SNR:int, *, show=True, path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # Setting the coordinate grid
    _x = np.arange(q)
    _y = np.arange(q)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    # Ravelling 2d mtx to 1d && setting bar params
    mtx_1d = mtx.ravel()
    bottom = np.zeros_like(mtx_1d)
    width = depth = 0.8

    ax.bar3d(x, y, bottom, width, depth, mtx_1d, shade=True, edgecolor='black')
    ax.set_xticks(np.arange(q))
    ax.set_yticks(np.arange(q))
    ax.set_zticks(np.arange(0, 1, 0.1))
    
    ax.set_title(f'$\gamma$ = {SNR}', fontsize=24)
    ax.set_xlabel('i', style='italic', fontsize=18)
    ax.set_ylabel('l', style='italic', fontsize=18)
    ax.set_zlabel('$P_c$', fontsize=18)

    if path is not None:
        plt.savefig(f'{path}gamma{SNR}.png')
    if show is True:
        plt.show()


def print_2d(list_2d):
    [print(row) for row in list_2d]
