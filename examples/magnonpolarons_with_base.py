# coding: utf-8
# !/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
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
N = 2  # number of sites per unit cell
a = 12.376e-10  # interatomic distance

# phonon constants
hbar = 6.582e-13
m = 9.8e-24

phonon_speeds = np.array([7209, 3843, 3843])
Phi = m * np.diag(2*phonon_speeds/a)**2

# magnon constants
R = a * np.arange(0, N)
mu_B = 5.788e-2
g = 2
S = 20 * np.ones(N)
J = 0.24
Dz = 0
Bz = np.zeros(N)
Bz = 0.537 * np.ones(N)
# Bz = 0.1 * np.sin(2*np.pi/(N*a) * R)

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.zeros((N, 3, 3*N))
for j in range(N):
    Bprime[j,0,3*j] = 1  # d(Bx)/dx
    Bprime[j,1,3*j] = 0.0  # d(By)/dx
    Bprime[j,2,3*j] = 0.0  # d(Bz)/dx

    Bprime[j,0,3*j+1] = 0.0  # d(Bx)/dy
    Bprime[j,1,3*j+1] = 0.0  # d(By)/dy
    Bprime[j,2,3*j+1] = 0.0  # d(Bz)/dy

    Bprime[j,0,3*j+2] = 0.0  # d(Bx)/dz
    Bprime[j,1,3*j+2] = 0.0  # d(By)/dz
    Bprime[j,2,3*j+2] = 0.0  # d(Bz)/dz

# lattice constants
k_arr = np.linspace(-np.pi/(N*a), np.pi/(N*a), num=300)

k_arr = np.linspace(1.1e8, 1.8e8, num=100)
ylim = [1.2, 1.6]

# k_arr = np.linspace(0.99*np.pi/(N*a), np.pi/(N*a), num=100)
# ylim = [10.7, 11]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def show_matrix(X):
    plt.imshow(np.imag(X))
    plt.grid()
    plt.colorbar()
    plt.show()


def order_eig(X):
    vals, vecs = np.linalg.eigh(X)
    i_arr = np.argsort(vals)[::-1]
    return vals[i_arr], vecs[:, i_arr]


def Phi_matrix(k):
    """
    Fourier transform of elastic interactions
    """
    matrix = np.zeros((3*N, 3*N), dtype=complex)

    for j1 in range(0,N):
        for j2 in range(0,N):
            if(j1 == j2):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] += 2 * Phi
            if(j1-1 == j2):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi
            if(j1 == j2-1):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi
            if(j1 == 0 and j2 == N-1):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi * np.exp(-1j*k*N*a)
            if(j1 == N-1 and j2 == 0):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi * np.exp(1j*k*N*a)

    return matrix


def J_matrix(k):
    matrix = np.zeros((N, N), dtype=complex)

    for j1 in range(N):
        for j2 in range(N):
            if(j1-1 == j2):
                matrix[j1,j2] += J
            if(j1 == j2-1):
                matrix[j1,j2] += J
            if(j1 == 0 and j2 == N-1):
                matrix[j1,j2] += J * np.exp(-1j*k*N*a)
            if(j1 == N-1 and j2 == 0):
                matrix[j1,j2] += J * np.exp(1j*k*N*a)

    return matrix


def Dz_matrix(k):
    matrix = np.zeros((N, N), dtype=complex)

    for j1 in range(0,N):
        for j2 in range(0,N):
            if(j1-1 == j2):
                matrix[j1,j2] += Dz
            if(j1 == j2-1):
                matrix[j1,j2] -= Dz
            if(j1 == 0 and j2 == N-1):
                matrix[j1,j2] += Dz * np.exp(-1j*k*N*a)
            if(j1 == N-1 and j2 == 0):
                matrix[j1,j2] -= Dz * np.exp(1j*k*N*a)

    return matrix


def Dprime_tensor(k):
    pass


def create_Tmatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, c{-k,1...3*N}]
    Tmatrix = np.zeros((7*N, 7*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)
    Dprime_k, Dprime_0 = Dprime_tensor(k), Dprime_tensor(0)

    # magnon terms
    for j1 in range(N):
        Tmatrix[j1,j1] += mu_B*g*S[j1]*Bz[j1]
        for j2 in range(N):
            Tmatrix[j1,j1] += 0.5*S[j2] * J0[j1,j2]
            Tmatrix[j2,j2] += 0.5*S[j1] * J0[j1,j2]
            Tmatrix[j1,j2] -= 0.5*np.sqrt(S[j1]*S[j2]) * \
                (Jk[j1,j2] + np.conjugate(Jk[j2,j1]))
            Tmatrix[j1,j2] -= 0.5j*np.sqrt(S[j1]*S[j2]) * \
                (Dk[j1,j2] - np.conjugate(Dk[j2,j1]))

    # phonon terms
    for lambd in range(3*N):
        w = np.sqrt(phi[lambd] / m)
        Tmatrix[N+lambd,N+lambd] += hbar*w
        Tmatrix[4*N+lambd,4*N+lambd] += hbar*w

    # magnon-phonon interaction
    for j in range(N):
        for lambd in range(3*N):
            w = np.sqrt(phi[lambd] / m)
            Bprime_term = np.sum(-mu_B*g*np.sqrt(hbar*S[j]/(4*m*w)) * \
                (Bprime[j,0,:]+1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )

            Tmatrix[j,N+lambd] += Bprime_term
            Tmatrix[N+lambd,j] += np.conjugate(Bprime_term)

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, c{-k,1...3*N}]
    Umatrix = np.zeros((7*N, 7*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)

    # magnon terms
    None

    # phonon terms
    None

    # magnon-phonon interaction
    for lambd in range(3*N):
        for j in range(N):
            w = np.sqrt(phi[lambd] / m)
            Bprime_term = np.sum(-mu_B*g*np.sqrt(hbar*S[j]/(4*m*w)) * \
                (Bprime[j,0,:]+1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )

            Umatrix[j,4*N+lambd] += Bprime_term
            Umatrix[4*N+lambd,j] += Bprime_term

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((14*N, num_k), dtype=complex)
    z = np.zeros((14*N, num_k))
    U = np.zeros((num_k,14*N,14*N), dtype=complex)
    i_magnons = np.concatenate([np.arange(0,N), np.arange(7*N,8*N)])

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i], U[i,:,:] = bosonic_system.diagonalize()
        for j in range(energies.shape[0]):
            z[j,i] = np.sum(np.abs(U[i,i_magnons,j])**2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(len(energies)):
        x, y = k_arr, np.real(energies[j,:])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(vmin=0, vmax=1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(z[j,:])
        line = ax.add_collection(lc)

    ax.set_xlim(k_arr.min(), k_arr.max())
    try:
        ax.set_ylim(ylim)
    except:
        ax.set_ylim(0, np.real(energies).max())

    ax.grid()
    cbar = fig.colorbar(line, ax=ax, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["Ph", "M"])
    fig.tight_layout()
    fig.savefig("img/magnon_polarons_with_base.png")

    # for j in range(3):
    #     plt.plot(k_arr, phonon_energy(k_arr,j,True), "blue")
    #     plt.plot(k_arr, phonon_energy(k_arr,j), "red")
    plt.show()
    plt.close(fig)


def phonon_energy(k, j=0, linear_dispersion=False):
    if(linear_dispersion is True):
        return hbar * phonon_speeds[j] * k
    else:
        return hbar*np.sqrt(Phi[j,j]/m) * np.abs(np.sin(0.5*k*a))

def magnon_energy(k, j=0):
    return 2*J*S[j]*(1-np.cos(k*a)) - 2*Dz*S[j]*np.sin(k*a) + Bz[j]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
