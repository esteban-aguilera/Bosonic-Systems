# coding: utf-8
# !/usr/bin/env python

import numpy as np
import os
import sys

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, "..")
from BosonicSystem import *

os.chdir("..")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Constants
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# phonon constants
hbar = 1
m = 1
omega = 3

# magnon constants
mu_B = 1
g = 1
S = 1
J = 1
Dz = 0.0
Bz = 0

# magnon polarons constants
Dprime = [0.0, 0.0, 0.0]
Bprime = [.05, 0.0, 0.0]

# lattice constants
a = 1
k_arr = np.linspace(-np.pi/a, np.pi/a, num=1000)

k_arr = np.linspace(1.5, 1.9, num=100)
ylim = [2.0, 2.5]

klim = [k_arr[0], k_arr[-1]]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def psi(k):
    return np.sqrt(0.25*hbar*S/(m*omega*np.abs(np.sin(0.5*k*a)))) * (
        S*(1j*Dprime[0]+Dprime[1])*np.abs(np.exp(1j*k*a)-1)**2 -
        mu_B*g*(Bprime[0]+1j*Bprime[1])
    )

def create_Tmatrix(k):
    # alpha = {a{k}, c{k}, a{-k}, c{-k}}
    Tmatrix = np.zeros((4, 4), dtype=complex)

    Tmatrix[0,0] += 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz
    Tmatrix[1,1] += hbar*omega*np.abs(np.sin(0.5*-k*a))
    Tmatrix[2,2] += 2*J*S*(1-np.cos(-k*a)) - 2*Dz*S*np.sin(-k*a) + mu_B*g*Bz
    Tmatrix[3,3] += hbar*omega*np.abs(np.sin(0.5*-k*a))

    Tmatrix[0,1] += psi(k)
    Tmatrix[1,0] += np.conjugate(psi(k))

    Tmatrix[2,3] += psi(k)
    Tmatrix[3,2] += psi(k)

    return Tmatrix


def create_Umatrix(k):
    # alpha = {a{k}, c{k}, a{-k}, c{-k}}
    Umatrix = np.zeros((4, 4), dtype=complex)

    Umatrix[0,3] += 0.5*psi(k)
    Umatrix[3,0] += 0.5*psi(k)

    Umatrix[1,2] += 0.5*psi(k)
    Umatrix[2,1] += 0.5*psi(k)

    return Umatrix


def dispersion_relation(k_arr):
    global Bprime

    num_k = k_arr.shape[0]
    dim = 8

    energies = np.zeros((num_k, dim), dtype=complex)
    U = np.zeros((num_k,dim,dim), dtype=complex)
    z = np.zeros((num_k,dim))

    for i, k in enumerate(k_arr):
        if(-1 < k and k < 1):
            Bprime = np.array( [0.0, 0.0, 0.0] )
        else:
            Bprime = np.array( [0.05, 0.0, 0.0] )
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[i,:], U[i,:,:] = bosonic_system.diagonalize()
        for j in range(dim):
            z[i,j] = np.abs(U[i,0,j])**2 + np.abs(U[i,2,j])**2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(dim):
        x, y = k_arr[:], np.real(energies[:,j])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(vmin=0, vmax=1)
        lc = LineCollection(segments, cmap='brg', norm=norm)
        lc.set_array(z[:,j])
        line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["Ph", "M"])

    ax.grid()
    ax.set_xlim(klim)
    try:
        ax.set_ylim(ylim)
    except:
        ax.set_ylim(0, np.real(energies).max())

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
