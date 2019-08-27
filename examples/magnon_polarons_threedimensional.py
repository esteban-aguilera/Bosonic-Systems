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
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

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
def psi(k, eps_k):
    s = 0
    for beta in range(3):
        s += np.sqrt(0.25*hbar*S) * (
            S*(1j*Dprime[0,beta]+Dprime[1,beta]) * np.abs(np.exp(1j*k*a)-1)**2 -
            mu_B*g*(Bprime[0,beta]-1j*Bprime[1,beta])
        )*eps_k[beta]
    return s


def create_Tmatrix(k):
    # alpha = [a{k}, c{k,1}, c{k,2}, c{k,3}, a{-k}, c{-k,1}, c{-k,2}, c{-k,3}]
    Tmatrix = np.zeros((8, 8), dtype=complex)

    vals, vecs = np.linalg.eig(Omega)
    u = hbar/(4*m)
    v = hbar/4 * (np.exp(1j*k*a)-1)*(np.exp(-1j*k*a)-1) * vals
    w = 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz

    # magnon terms
    Tmatrix[0,0] = w
    Tmatrix[4,4] = w

    # phonon terms
    for lambd in range(1,4):
        Tmatrix[lambd,lambd] = u + v[lambd-1]
        Tmatrix[4+lambd,4+lambd] = u + v[lambd-1]

    # magnon-phonon interaction

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k}, c{k,1}, c{k,2}, c{k,3}, a{-k}, c{-k,1}, c{-k,2}, c{-k,3}]
    Umatrix = np.zeros((8, 8), dtype=complex)

    for lambd in range(1,4):  # loop over phonon operators
        pass

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((8, num_k))
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
    fig.savefig("img/magnon_polarons_threedimensional.png")

    plt.show()
    plt.close(fig)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
