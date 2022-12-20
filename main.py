import numpy as np

from consts import SNRs, q
from utils import calc_transition_prob, render_3d, print_2d


def main():

    for SNR in SNRs:

        mtx = np.zeros((q,q))
        P_e = calc_transition_prob(SNR)

        for i in range(q):
            for l in range(q):
                if l == i:
                    mtx[i][l] = 1 - P_e
                else:
                    mtx[i][l] = P_e / (q - 1)

        render_3d(mtx, SNR, show=True, path='3d-plots/')
        # print_2d(mtx)
        # print()


if __name__ == "__main__":
    main()
    