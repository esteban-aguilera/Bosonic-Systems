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
Phi = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 3.0]])
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
def order_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(vals)

    return vals[i_arr], vecs[:, i_arr]


def Phi_matrix(k):
    matrix = np.zeros((3*N, 3*N), dtype=complex)

    for j1 in range(0,3*N,3):
        for j2 in range(0,3*N,3):
            if(j1 == j2):
                matrix[j1:j1+3, j2:j2+3] += 2 * Phi
            if(j1-3 == j2):
                matrix[j1:j1+3, j2:j2+3] -= Phi
            if(j1 == j2-3):
                matrix[j1:j1+3, j2:j2+3] -= Phi
            if(j1 == 0 and j2 == 3*N-3):
                matrix[j1:j1+3, j2:j2+3] -= Phi * np.exp(-1j*k*N*a)
            if(j1 == 3*N-3 and j2 == 0):
                matrix[j1:j1+3, j2:j2+3] -= Phi * np.exp(1j*k*N*a)

    return matrix / m


def create_Tmatrix(k):
    Tmatrix = np.zeros((2*3*N, 2*3*N), dtype=complex)
    phi, eps_arr = order_eig(Phi_matrix(-k))

    for lambd in range(3*N):
        Tmatrix[lambd,lambd] = phi[lambd] + 1
        Tmatrix[3*N+lambd,3*N+lambd] = phi[lambd] + 1

    return hbar/4.0 * Tmatrix


def create_Umatrix(k):
    Umatrix = np.zeros((2*3*N, 2*3*N), dtype=complex)
    phi, eps_arr = order_eig(Phi_matrix(-k))

    for lambd in range(3*N):
        Umatrix[lambd,3*N+lambd] = phi[lambd] - 1
        Umatrix[3*N+lambd,lambd] = phi[lambd] - 1

    return hbar/4.0 * Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*3*N, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(3):
        ax.plot(k_arr, analytical_energies(k_arr, j), color="darkblue")
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/phonons_with_base.png")

    plt.show()
    plt.close(fig)


def analytical_energies(k, j=0):
    return hbar*np.sqrt(Phi[j,j])*np.abs(np.sin(0.5*k_arr*a))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
