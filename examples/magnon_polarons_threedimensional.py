# coding: utf-8
# !/usr/bin/env python

import numpy as np
import os
import sys

sys.path.insert(0, "..")
from BosonicSystem import *

os.chdir("..")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Constants
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# phonon constants
hbar = 1
m = 1
Omega = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 3.0]])

# magnon constants
mu_B = 1
g = 1
S = 1
J = 1
Dz = 0.0
Bz = 0

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])

# lattice constants
a = 1
k_arr = np.linspace(-np.pi/a, np.pi/a, num=100)

k_arr = np.linspace(0.45, 0.55, num=100)
# ylim = [0.20, 0.3]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_Tmatrix(k):
    # alpha = [a{k}, c{k,1}, c{k,2}, c{k,3}, a{-k}, c{-k,1}, c{-k,2}, c{-k,3}]
    Tmatrix = np.zeros((8, 8), dtype=complex)

    for lambd in range(1,4):  # loop over phonon operators
        pass

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k}, c{k,1}, c{k,2}, c{k,3}, a{-k}, c{-k,1}, c{-k,2}, c{-k,3}]
    Umatrix = np.zeros((8, 8), dtype=complex)

    for lambd in range(1,4):  # loop over phonon operators
        pass

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*3, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/threedimensional_phonons.png")

    plt.show()
    plt.close(fig)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
