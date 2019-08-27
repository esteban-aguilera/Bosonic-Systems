# coding: utf-8
# !/usr/bin/env python

import numpy as np
import os
import sys

sys.path.insert(0, "..")
from BosonicSystem import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Constants
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
hbar = 1
m = 1
omega = np.array( [1, 2, 3] )
a = 1
N = 2
k_arr = np.linspace(-np.pi/(N*a), np.pi/(N*a), num=100)

os.chdir("..")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def phi_matrix(k):
    phi = np.zeros((3*N, 3*N), dtype=complex)

    for j1 in range(0,3*N,3):
        for j2 in range(0,3*N,3):
            if(j1 == j2):
                phi[j1:j1+3, j2:j2+3] += 2 * np.diag( omega )
            if(j1-3 == j2):
                phi[j1:j1+3, j2:j2+3] -= np.diag( omega )
            if(j1 == j2-3):
                phi[j1:j1+3, j2:j2+3] -= np.diag( omega )
            if(j1 == 0 and j2 == 3*N-3):
                phi[j1:j1+3, j2:j2+3] -= np.diag( omega ) * np.exp(-1j*k*N*a)
            if(j1 == 3*N-3 and j2 == 0):
                phi[j1:j1+3, j2:j2+3] -= np.diag( omega ) * np.exp(1j*k*N*a)

    return phi / m

def create_Tmatrix(phi):
    Tmatrix = np.zeros((2*3*N, 2*3*N), dtype=complex)
    for lambd in range(3*N):
        Tmatrix[lambd,lambd] = phi[lambd] + 1
        Tmatrix[3*N+lambd,3*N+lambd] = phi[lambd] + 1

    return hbar/4 * Tmatrix


def create_Umatrix(phi):
    Umatrix = np.zeros((2*3*N, 2*3*N), dtype=complex)
    for lambd in range(3*N):
        Umatrix[lambd,3*N+lambd] = phi[lambd] - 1
        Umatrix[3*N+lambd,lambd] = phi[lambd] - 1

    return hbar/4 * Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*3*N, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        phi = np.sort( np.linalg.eigvals( phi_matrix(-k) ) )
        bosonic_system = BosonicSystem(create_Tmatrix(phi), create_Umatrix(phi))
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/phonons_with_base.png")

    plt.show()
    plt.close(fig)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
