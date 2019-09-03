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
a = 1  # interatomic distance

# phonon constants
hbar = 1
m = 1 * np.ones(N)
Phi = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0]])

# magnon constants
R = a * np.arange(0, N)
mu_B = 1
g = 1
S = 1 * np.ones(N)
J = 1
Dz = 0
Bz = np.zeros(N)
Bz = np.linspace(0,1,num=N)
Bz = np.sin(2*np.pi/(N*a) * R)

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.zeros((N, 3, 3*N))
for j in range(N):
    Bprime[j,0,3*j] = 0.01
    Bprime[j,1,3*j+1] = 0.02
    Bprime[j,2,3*j] = 0.03

# lattice constants
k_arr = np.linspace(-np.pi/(N*a), np.pi/(N*a), num=300)

k_arr = np.linspace(0.46, 1.1, num=100)
ylim = [0.3, 1]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def show_matrix(X):
    plt.imshow(np.abs(X))
    plt.grid()
    plt.colorbar()
    plt.show()


def order_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(vals)

    return vals[i_arr], vecs[:, i_arr]


def Phi_matrix(k):
    """
    Fourier transform of elastic interactions
    """
    matrix = np.zeros((3*N, 3*N), dtype=complex)

    for j1 in range(0,N):
        for j2 in range(0,N):
            if(j1 == j2):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] += 2 * Phi / np.sqrt(m[j1]*m[j2])
            if(j1-1 == j2):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi / np.sqrt(m[j1]*m[j2])
            if(j1 == j2-1):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi / np.sqrt(m[j1]*m[j2])
            if(j1 == 0 and j2 == N-1):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi / np.sqrt(m[j1]*m[j2]) * \
                    np.exp(-1j*k*N*a)
            if(j1 == N-1 and j2 == 0):
                matrix[3*j1:3*j1+3, 3*j2:3*j2+3] -= Phi / np.sqrt(m[j1]*m[j2]) * \
                    np.exp(1j*k*N*a)

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
    # alpha = [a{k,1...N}, c{k,1...3*N}, a{-k,1...N}, c{-k,1...3*N}]
    Tmatrix = np.zeros((8*N, 8*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(-k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)
    Dprime_k, Dprime_0 = Dprime_tensor(k), Dprime_tensor(0)

    # magnon terms
    for j1 in range(N):
        Tmatrix[j1,j1] += mu_B*g*S[j1]*Bz[j1]
        Tmatrix[4*N+j1,4*N+j1] += mu_B*g*S[j1]*Bz[j1]
        for j2 in range(N):
            Tmatrix[j1,j1] += 0.5*S[j2] * J0[j1,j2]
            Tmatrix[j2,j2] += 0.5*S[j1] * J0[j1,j2]
            Tmatrix[j1,j2] -= 0.5*np.sqrt(S[j1]*S[j2]) * \
                (Jk[j1,j2] + np.conjugate(Jk[j2,j1]))
            Tmatrix[j1,j2] -= 0.5j*np.sqrt(S[j1]*S[j2]) * \
                (Dk[j1,j2] - np.conjugate(Dk[j2,j1]))

            Tmatrix[4*N+j1,4*N+j1] += 0.5*S[j2] * J0[j1,j2]
            Tmatrix[4*N+j2,4*N+j2] += 0.5*S[j1] * J0[j1,j2]
            Tmatrix[4*N+j1,4*N+j2] -= 0.5*np.sqrt(S[j1]*S[j2]) * \
                (np.conjugate(Jk[j1,j2]) + Jk[j2,j1])
            Tmatrix[4*N+j1,4*N+j2] -= 0.5j*np.sqrt(S[j1]*S[j2]) * \
                (np.conjugate(Dk[j1,j2]) - Dk[j2,j1])

    # phonon terms
    for lambd in range(3*N):
        Tmatrix[N+lambd,N+lambd] += 0.25*hbar * (np.conjugate(phi[lambd]) + 1)
        Tmatrix[5*N+lambd,5*N+lambd] += 0.25*hbar * (np.conjugate(phi[lambd]) + 1)

    # magnon-phonon interaction
    for beta in range(3*N):
        for lambd in range(3*N):
            for j1 in range(N):
                Tmatrix[j1,N+lambd] -= 0.5 * mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    (Bprime[j1,0,beta]+1j*Bprime[j1,1,beta])*eps_arr[beta,lambd]
                Tmatrix[N+lambd,j1] -= 0.5 * mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    np.conjugate((Bprime[j1,0,beta]+1j*Bprime[j1,1,beta]) *
                                 eps_arr[beta,lambd])
                Tmatrix[5*N+lambd,4*N+j1] -= 0.5 * mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    (Bprime[j1,0,beta]-1j*Bprime[j1,1,beta])*eps_arr[beta,lambd]
                Tmatrix[4*N+j1,5*N+lambd] -= 0.5 * mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    np.conjugate((Bprime[j1,0,beta]-1j*Bprime[j1,1,beta]) *
                                 eps_arr[beta,lambd])
                for j2 in range(N):
                    pass

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, a{-k,1...N}, c{-k,1...3*N}]
    Umatrix = np.zeros((8*N, 8*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)

    # magnon terms
    None

    # phonon terms
    for lambd in range(3*N):
        Umatrix[N+lambd,5*N+lambd] += 0.25*hbar * (np.conjugate(phi[lambd]) - 1)
        Umatrix[5*N+lambd,N+lambd] += 0.25*hbar * (np.conjugate(phi[lambd]) - 1)

    # magnon-phonon interaction
    for beta in range(3*N):
        for lambd in range(3*N):
            for j1 in range(N):
                Umatrix[j1,5*N+lambd] -= mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    (Bprime[j1,0,beta]+1j*Bprime[j1,1,beta])*eps_arr[beta,lambd]
                Umatrix[N+lambd,4*N+j1] -= mu_B*g*np.sqrt(0.25*hbar*S[j1]) * \
                    np.conjugate((Bprime[j1,0,beta]-1j*Bprime[j1,1,beta]) * \
                                 eps_arr[beta,lambd])

    # show_matrix(Umatrix)
    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*8*N, num_k), dtype=complex)
    z = np.zeros((16*N, num_k))
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i], U[i] = bosonic_system.diagonalize()
        for j in range(len(energies[:,i])):
            for k in range(4*N):
                prob = np.sum(np.abs(U[i][0:N,j])**2) + np.sum(np.abs(U[i][4*N:5*N,j])**2)
                if(prob > z[j,i]):
                    z[j,i] = prob

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(len(energies)//2):
        x, y = k_arr, energies[j,:]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(z.min(), z.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(z[j,:])
        line = ax.add_collection(lc)

    ax.set_xlim(k_arr.min(), k_arr.max())
    try:
        ax.set_ylim(ylim)
    except:
        ax.set_ylim(0, energies.max())

    ax.grid()
    fig.colorbar(line, ax=ax)
    fig.tight_layout()
    fig.savefig("img/magnon_polarons_with_base.png")

    plt.show()
    plt.close(fig)


def phonon_energy(k, j=0):
    return hbar*np.sqrt(Phi[j,j])*np.abs(np.sin(0.5*k_arr*a))

def magnon_energy(k, j=0):
    return 2*J*S[j]*(1-np.cos(k*a)) - 2*Dz*S[j]*np.sin(k*a) + Bz[j]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
