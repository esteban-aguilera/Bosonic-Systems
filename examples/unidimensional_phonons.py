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
omega = 1
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
    Tmatrix = np.zeros((2, 2), dtype=complex)

    # alpha = {a{k}, a{-k}}
    Tmatrix[0,0] = hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2
    Tmatrix[1,1] = hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2

    return Tmatrix


def create_Umatrix(k):
    Umatrix = np.zeros((2, 2), dtype=complex)

    # alpha{k} = {a{k}, a{-k}}
    Umatrix[0,1] = -hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2
    Umatrix[1,0] = -hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        # energies[:,i], U[i] = bosonic_system.diagonalize()
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_arr, analytical_energies(k), color="darkblue")
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/unidimensional_phonons.png")

    plt.show()
    plt.close(fig)


def analytical_energies(k):
    return hbar*omega*np.abs(np.sin(0.5*k_arr*a))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
