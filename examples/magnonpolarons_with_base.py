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
N = 1  # number of sites per unit cell

# phonon constants
hbar = 1
m = 1 * np.ones(N)
Omega = np.array([[2.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

# magnon constants
mu_B = 1
g = 1
S = 1 * np.ones(N)
J = 1
Dz = 0
Bz = 0 * np.ones(N)

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.zeros((N, 3, 3*N))
for j in range(N):
    Bprime[j,0,3*j] = 0.01
    Bprime[j,1,3*j] = 0.02
    Bprime[j,2,3*j] = 0.03

# lattice constants
a = 1
k_arr = np.linspace(-np.pi/(N*a), np.pi/(N*a), num=100)

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
def order_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(vals)

    return vals[i_arr], vecs[:, i_arr]


def Phi_matrix(k):
    """
    Fourier transform of elastic interactions
    """
    matrix = np.zeros((3*N, 3*N), dtype=complex)

    for j1 in range(0,3*N,3):
        for j2 in range(0,3*N,3):
            if(j1 == j2):
                matrix[j1:j1+3, j2:j2+3] += 2 * Omega / np.sqrt(m[j1//3]*m[j2//3])
            if(j1-3 == j2):
                matrix[j1:j1+3, j2:j2+3] -= Omega / np.sqrt(m[j1//3]*m[j2//3])
            if(j1 == j2-3):
                matrix[j1:j1+3, j2:j2+3] -= Omega / np.sqrt(m[j1//3]*m[j2//3])
            if(j1 == 0 and j2 == 3*N-3):
                matrix[j1:j1+3, j2:j2+3] -= Omega / np.sqrt(m[j1//3]*m[j2//3]) * \
                    np.exp(-1j*k*N*a)
            if(j1 == 3*N-3 and j2 == 0):
                matrix[j1:j1+3, j2:j2+3] -= Omega / np.sqrt(m[j1//3]*m[j2//3]) * \
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


def Psi(k, eps_k):
    s = 0
    for beta in range(3):
        s += np.sqrt(0.25*hbar*S) * (
            S*np.abs(np.exp(1j*k*a)-1)**2*(1j*Dprime[0,beta]+Dprime[1,beta]) -
            mu_B*g*(Bprime[0,beta]-1j*Bprime[1,beta])
        )*eps_k[beta]
    return s


def create_Tmatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, a{-k,1...N}, c{-k,1...3*N}]
    Tmatrix = np.zeros((8*N, 8*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)
    Dprime_k, Dprime_0 = Dprime_tensor(k), Dprime_tensor(0)

    # magnon terms
    for j1 in range(N):
        for j2 in range(N):
            Tmatrix[j1,j1] += 0.5*np.sqrt(S[j1]*S[j2]) * J0[j1,j2]
            Tmatrix[j2,j2] += 0.5*np.sqrt(S[j1]*S[j2]) * J0[j1,j2]
            Tmatrix[j1,j2] -= 0.5*np.sqrt(S[j1]*S[j2]) * \
                (Jk[j1,j2] + np.conjugate(Jk[j1,j2]))
            Tmatrix[j1,j2] -= 0.5j*np.sqrt(S[j1]*S[j2]) * \
                (Dk[j1,j2] - np.conjugate(Dk[j1,j2]))
            Tmatrix[j1,j2] += Bz[j1] * (j1 == j2)

            Tmatrix[4*N+j1,4*N+j1] += 0.5*np.sqrt(S[j1]*S[j2]) * J0[j1,j2]
            Tmatrix[4*N+j2,4*N+j2] += 0.5*np.sqrt(S[j1]*S[j2]) * J0[j1,j2]
            Tmatrix[4*N+j1,4*N+j2] -= 0.5*np.sqrt(S[j1]*S[j2]) * \
                (Jk[j1,j2] + np.conjugate(Jk[j1,j2]))
            Tmatrix[4*N+j1,4*N+j2] -= 0.5j*np.sqrt(S[j1]*S[j2]) * \
                (Dk[j1,j2] - np.conjugate(Dk[j1,j2]))
            Tmatrix[4*N+j1,4*N+j2] += Bz[j1] * (j1 == j2)

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
                Umatrix[j1,5*N+lambd] -= mu_B*g*np.sqrt(0.25*hbar*S) * \
                    (Bprime[j1,0,beta]+1j*Bprime[j1,1,beta])*eps_arr[beta,lambd]
                Umatrix[N+lambd,4*N+j1] -= mu_B*g*np.sqrt(0.25*hbar*S) * \
                    np.conjugate((Bprime[j1,0,beta]-1j*Bprime[j1,1,beta]) * \
                                 eps_arr[beta,lambd])

    return Umatrix


def dispersion_relation(k_arr):
    num_k = k_arr.shape[0]

    energies = np.zeros((2*8*N, num_k), dtype=complex)
    U = [None for _ in range(num_k)]

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i], U[i] = bosonic_system.diagonalize()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(3):
        ax.plot(k_arr, phonon_energy(k_arr, j), color="darkblue")
    ax.plot(k_arr, magnon_energy(k_arr), color="darkblue")
    for i in range(len(energies)//2):
        ax.plot(k_arr, np.real(energies[i,:]), '--', color="orange")

    try:
        ax.set_ylim(ylim)
    except:
        pass
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/magnon_polarons_with_base.png")

    plt.show()
    plt.close(fig)


def phonon_energy(k, j=0):
    return hbar*Omega[j,j]*np.abs(np.sin(0.5*k_arr*a))

def magnon_energy(k):
    return 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + Bz


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
