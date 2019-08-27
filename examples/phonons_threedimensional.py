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
phi = np.array( [1, 2, 3] )
a = 1
k_arr = np.linspace(-np.pi/a, np.pi/a, num=100)

os.chdir("..")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_Tmatrix(k):
    u = hbar/(4*m)
    v = hbar/4 * (np.exp(1j*k*a)-1)*(np.exp(-1j*k*a)-1) * phi
    Tmatrix = np.zeros((2*3, 2*3), dtype=complex)

    # alpha = {c{k,1}, c{k,2}, c{k,3}, c{-k,1}, c{-k,2}, c{-k,3}}
    for lambd in range(3):
        Tmatrix[lambd,lambd] = u + v[lambd]
        Tmatrix[3+lambd,3+lambd] = u + v[lambd]

    return Tmatrix


def create_Umatrix(k):
    u = hbar/(4*m)
    v = hbar/4 * (np.exp(1j*k*a)-1)*(np.exp(-1j*k*a)-1) * phi
    Umatrix = np.zeros((2*3, 2*3), dtype=complex)

    # alpha = {a{k}, a{-k}}
    for lambd in range(3):
        Umatrix[lambd,3+lambd] = -u + v[lambd]
        Umatrix[3+lambd,lambd] = -u + v[lambd]

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*3, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        # energies[:,i], U[i] = bosonic_system.diagonalize()
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/phonons_threedimensional.png")

    plt.show()
    plt.close(fig)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
