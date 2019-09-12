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
a = 12.376e-10  # interatomic distance

# phonon constants
hbar = 6.582e-13
m = 9.8e-24
phonon_speeds = np.array([3843, 3843, 7209])
Omega = m * np.diag(2*phonon_speeds/a)**2

# magnon constants
mu_B = 5.788e-2
g = 2
S = 20
J = 0.24
Dz = 0
Bz = 0.3

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])

# lattice constants
k_arr = np.linspace(-np.pi/a, np.pi/a, num=1000)
k_arr = np.linspace(0.2e9, np.pi/a, num=1000)


# k_arr = np.linspace(1.42, 1.56, num=1000) * 1e9
# ylim = [11.8, 12.6]

klim = [k_arr[0], k_arr[-1]]

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

    phi, eps_arr = order_eig(Omega)

    # magnon terms
    Tmatrix[0,0] += 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz
    Tmatrix[4,4] += 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz

    # phonon terms
    for lambd in range(1,4):
        w = np.sqrt(phi[lambd-1] / m) * np.abs(np.exp(1j*k*a)-1)
        Tmatrix[lambd,lambd] += hbar*w
        Tmatrix[4+lambd,4+lambd] += hbar*w

    # magnon-phonon interaction
    for lambd in range(1,4):
        w = np.sqrt(phi[lambd-1] / m) * np.abs(np.exp(1j*k*a)-1)
        Bterm = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
            (Bprime[0,:]+1j*Bprime[1,:])*eps_arr[:,lambd-1]
        )
        Bterm_c = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
            (Bprime[0,:]+1j*Bprime[1,:])*np.conjugate(eps_arr[:,lambd-1])
        )

        Tmatrix[0,lambd] += Bterm
        Tmatrix[lambd,0] += np.conjugate(Bterm)

        Tmatrix[4,4+lambd] += Bterm_c
        Tmatrix[4+lambd,4] += np.conjugate(Bterm_c)

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k}, c{k,1}, c{k,2}, c{k,3},  a{-k}, c{-k,1}, c{-k,2}, c{-k,3}]
    Umatrix = np.zeros((8, 8), dtype=complex)

    phi, eps_arr = order_eig(Omega)

    # magnon terms
    None

    # phonon terms
    None

    # magnon-phonon interaction
    for lambd in range(1,4):
        w = np.sqrt(phi[lambd-1] / m) * np.abs(np.exp(1j*k*a)-1)
        Bterm = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
            (Bprime[0,:]+1j*Bprime[1,:])*eps_arr[:,lambd-1]
        )
        Bterm_c = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
            (Bprime[0,:]+1j*Bprime[1,:])*np.conjugate(eps_arr[:,lambd-1])
        )

        Umatrix[0,4+lambd] += 0.5*Bterm
        Umatrix[4+lambd,0] += 0.5*Bterm

        Umatrix[4,lambd] += 0.5*Bterm_c
        Umatrix[lambd,4] += 0.5*Bterm_c

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]
    dim = 16

    energies = np.zeros((num_k, dim), dtype=complex)
    U = np.zeros((num_k,dim,dim), dtype=complex)
    z = np.zeros((num_k,dim))

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[i,:], U[i,:,:] = bosonic_system.diagonalize()
        # print(np.real(energies[i,:]), "\n\n")
        # show_matrix(U[i,:,:])
        for j in range(dim):
            z[i,j] = np.abs(U[i,0,j])**2 + np.abs(U[i,4,j])**2 + \
                np.abs(U[i,8,j])**2 + np.abs(U[i,12,j])**2

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

    ax.plot(k_arr, phonon_energy(k_arr), color="black")
    ax.plot(k_arr, magnon_energy(k_arr), color="black")
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


def show_matrix(X):
    plt.imshow(np.abs(X)**2)
    plt.grid()
    plt.colorbar()
    plt.show()


def order_eig(X):
    vals, vecs = np.linalg.eig(Omega)
    i_arr = np.argsort(vals)

    return vals[i_arr], vecs[:, i_arr]


def phonon_energy(k, j=2):
    return 2*hbar*np.sqrt(Omega[j,j]/m)*np.abs(np.sin(0.5*k_arr*a))

def magnon_energy(k):
    return 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
