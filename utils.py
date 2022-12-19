import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import interactive

from consts import q


def inner_integrand(u: np.float64) -> np.float64:
    return np.exp(- u*u / 2)


def integrand(z: np.float64, gamma: int) -> np.float64:
    inner_integrand_res = quad(lambda u: inner_integrand(u), z, np.inf)[0]
    return (1 - 1 / np.sqrt(2*np.pi) * inner_integrand_res)**(q-1) * \
        1 / np.sqrt(2*np.pi) * np.exp(-((z-np.sqrt(2*gamma))**2) / 2)


def calc_transition_prob(SNR: int) -> np.float64:
    gamma = SNR  # E/N_0
    # eta = np.random.normal(0, 1)
    # z = np.sqrt(2 * gamma) + eta
    return 1 - quad(lambda z: integrand(z, gamma), -np.inf, np.inf)[0]


def render_3d(mtx: np.ndarray, SNR:int, *, path=None):
    fig = plt.figure()
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

    ax.bar3d(x, y, bottom, width, depth, mtx_1d, shade=True)
    ax.set_title(f'$\gamma$ = {SNR}')

    if path is not None:
        plt.savefig(f'{path}gamma{SNR}.png')
    plt.show()

def print_2d(list_2d):
    [print(row) for row in list_2d]
