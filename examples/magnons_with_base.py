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
N = 2


S = np.ones(N)
J = 1
Dz = 0
Bz = np.linspace(0, 1, num=N)

a = 1
k_arr = np.linspace(-np.pi/(N*a), np.pi/(N*a), num=100)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    dispersion_relation(k_arr)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def Jq(k, j1, j2):
    sum = 0
    if(j1-1 == j2):
        sum += 1
    if(j1 == j2-1):
        sum += 1
    if(j1 == 0 and j2 == N-1):
        sum += np.exp(-1j*k*(N*a))
    if(j1 == N-1 and j2 == 0):
        sum += np.exp(1j*k*(N*a))

    return J*sum

def Dq(k, j1, j2):
    sum = 0
    if(j1-1 == j2):
        sum += 1
    if(j1 == j2-1):
        sum -= 1
    if(j1 == 0 and j2 == N-1):
        sum += np.exp(-1j*k*(N*a))
    if(j1 == N-1 and j2 == 0):
        sum -= np.exp(1j*k*(N*a))
    return Dz*sum

def create_Tmatrix(k):
    Tmatrix = np.zeros((2*N, 2*N), dtype=complex)

    for j1 in range(N):
        for j2 in range(N):
            Tmatrix[j1,j1] += 0.5*np.sqrt(S[j1]*S[j2])*Jq(0,j1,j2)
            Tmatrix[j2,j2] += 0.5*np.sqrt(S[j1]*S[j2])*Jq(0,j1,j2)
            Tmatrix[j1,j2] -= 0.5*np.sqrt(S[j1]*S[j2])*(Jq(k,j1,j2) + Jq(-k,j2,j1))
            Tmatrix[j1,j2] -= 0.5j*np.sqrt(S[j1]*S[j2])*(Dq(k,j1,j2) - Dq(-k,j2,j1))
            Tmatrix[j1,j2] += Bz[j1]*(j1 == j2)

            Tmatrix[N+j1,N+j1] += 0.5*np.sqrt(S[j1]*S[j2])*Jq(0,j1,j2)
            Tmatrix[N+j2,N+j2] += 0.5*np.sqrt(S[j1]*S[j2])*Jq(0,j1,j2)
            Tmatrix[N+j1,N+j2] -= 0.5*np.sqrt(S[j1]*S[j2])*(Jq(k,j1,j2) + Jq(-k,j2,j1))
            Tmatrix[N+j1,N+j2] -= 0.5j*np.sqrt(S[j1]*S[j2])*(Dq(k,j1,j2) - Dq(-k,j1,j2))
            Tmatrix[N+j1,N+j2] += Bz[j1]*(j1 == j2)

    return Tmatrix


def create_Umatrix(k):
    Umatrix = np.zeros((2*N, 2*N), dtype=complex)

    return Umatrix


def dispersion_relation(k_arr):
    energies = np.zeros((2*N, k_arr.shape[0]))
    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[:,i] = bosonic_system.energies()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(k_arr, analytical_energies(k_arr), color="darkblue")
    # ax.plot(k_arr, analytical_energies(-k_arr), color="darkgreen")
    for i in range(len(energies)):
        ax.plot(k_arr, energies[i,:], '--', color="orange")
    ax.grid()
    fig.tight_layout()
    fig.savefig("img/magnons_with_base.png")

    plt.show()
    plt.close(fig)


def analytical_energies(k):
    return 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + Bz


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
