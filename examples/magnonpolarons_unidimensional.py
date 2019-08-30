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
omega = 1

# magnon constants
mu_B = 1
g = 1
S = 1
J = 1
Dz = 0.0
Bz = 0

# magnon polarons constants
Dprime = [0.01, 0.02, 0.03]
Bprime = [0.01, 0.02, 0.03]

# lattice constants
a = 1
k_arr = np.linspace(-np.pi/a, np.pi/a, num=100)

k_arr = np.linspace(0.45, 0.55, num=100)
ylim = [0.20, 0.30]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def psi(k):
    return np.sqrt(0.25*hbar*S) * (
        S*(1j*Dprime[0]+Dprime[1])*np.abs(np.exp(1j*k*a)-1)**2 -
        mu_B*g*(Bprime[0]-1j*Bprime[1])
    )

def create_Tmatrix(k):
    # alpha = {a{k}, c{k}, a{-k}, c{-k}}
    Tmatrix = np.zeros((4, 4), dtype=complex)

    Tmatrix[0,0] = 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz
    Tmatrix[1,1] = hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2
    Tmatrix[2,2] = 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz
    Tmatrix[3,3] = hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2

    Tmatrix[0,1] = 0.5*np.conjugate(psi(k))
    Tmatrix[1,0] = 0.5*psi(k)

    Tmatrix[2,3] = 0.5*np.conjugate(psi(k))
    Tmatrix[3,2] = 0.5*psi(k)

    return Tmatrix


def create_Umatrix(k):
    # alpha = {a{k}, c{k}, a{-k}, c{-k}}
    Umatrix = np.zeros((4, 4), dtype=complex)

    Umatrix[1,3] = -hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2
    Umatrix[3,1] = -hbar/(4*m) + hbar*m*omega**2*(1-np.cos(k*a))/2

    Umatrix[0,3] = np.conjugate(psi(k))
    Umatrix[1,2] = np.conjugate(psi(k))

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((4*2, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i], U[i] = bosonic_system.diagonalize()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_arr, phonon_energy(k_arr), color="darkblue")
    ax.plot(k_arr, magnon_energy(k_arr), color="darkblue")
    for i in range(len(energies)//2):
        ax.plot(k_arr, energies[i,:], '--', color="orange")

    try:
        ax.set_ylim(ylim)
    except:
        pass
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/magnon_polarons_unidimensional.png")

    plt.show()
    plt.close(fig)


def phonon_energy(k):
    return hbar*omega*np.abs(np.sin(0.5*k_arr*a))

def magnon_energy(k):
    return 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + Bz


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
