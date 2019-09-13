# coding: utf-8
# !/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.integrate import simps

sys.path.insert(0, "..")
from BosonicSystem import *

os.chdir("..")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Constants
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
N = 1  # number of sites per unit cell
a = 12.376e-10  # interatomic distance

# phonon constants
hbar = 6.582e-13
m = 9.8e-24

phonon_speeds = np.array([3843, 3843, 7209])
Phi = m * np.diag(phonon_speeds/a)**2

# magnon constants
R = a * np.arange(0, N)
mu_B = 5.788e-2
g = 2
S = 20
J = 0.17
Dz = 0
Bz = np.zeros(N)
Bz = 0. * np.ones(N)  # 1.30

# magnon polarons constants
Dprime = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
Bprime = np.zeros((N, 3, 3*N)) + 1
for j in range(N):
    Bprime[j,0,3*j] += 0.0  # 2*np.pi/(N*a)*np.cos(2*np.pi/(N*a)*R[j])  # d(Bx)/dx
    Bprime[j,1,3*j] += 0.0  # d(By)/dx
    Bprime[j,2,3*j] += 0.0  # d(Bz)/dx

    Bprime[j,0,3*j+1] += 0.0 # d(Bx)/dy
    Bprime[j,1,3*j+1] += 0.0  # d(By)/dy
    Bprime[j,2,3*j+1] += 0.0  # d(Bz)/dy

    Bprime[j,0,3*j+2] += 0.0  # d(Bx)/dz
    Bprime[j,1,3*j+2] += 0.0  # d(By)/dz
    Bprime[j,2,3*j+2] += 0.0  # d(Bz)/dz

num_k = 1000
# 1Bz
klim = [-np.pi/(N*a), np.pi/(N*a)]

# positive Bz
klim = [0.3e9, 0.7e9]
ylim = [0,3]

# zoom
# klim = [1.15e9, 1.20e9]
# ylim = [11, 12]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    # dispersion_relation()
    # MPpresence()
    # plot_MPpresence("NvsBz_cos(x)")

    k = np.mean(klim)
    bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
    bosonic_system.plot_H()
    energies, U = bosonic_system.diagonalize()

    print(np.real(energies))
    show_matrix(np.abs(U[:8*N, :8*N]))

    # show_matrix(np.abs(create_Tmatrix(k)))
    # show_matrix(np.abs(create_Umatrix(k)))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Routines
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def dispersion_relation():
    k_arr = np.linspace(*klim, num=num_k)

    dim = 16*N
    dk = k_arr[1] - k_arr[0]

    energies = np.zeros((num_k,dim), dtype=complex)
    U = np.zeros((num_k,dim,dim), dtype=complex)
    z = np.zeros((num_k,dim))

    for i, k in enumerate(k_arr):
        bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
        energies[i,:], U[i,:,:] = bosonic_system.diagonalize()
        for j in range(dim):
            z[i,j] = np.sum(np.abs(U[i,:N,j])**2) + np.sum(np.abs(U[i,4*N:5*N,j])**2) + \
                np.sum(np.abs(U[i,8*N:9*N,j])**2) + np.sum(np.abs(U[i,12*N:13*N,j])**2)

    # plot dispersion relation
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
        ax.text(0.97*klim[1], 0.97*ylim[1], "$B_z=%.3f$",
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    except:
        ax.set_ylim(0, np.real(energies).max())
        ax.text(0.97*klim[1], 0.97*np.real(energies).max(), "$B_z=%.3f$" % Bz[0],
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))


    fig.tight_layout()
    fig.savefig("img/mp_dispersion.png")
    plt.show()
    plt.close(fig)

    # plot group velocity
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for j in range(dim):
    #     if(z[0,j] > 0.5):
    #         color="green"
    #     else:
    #         color="blue"
    #     plt.plot(k_arr[:-1], np.real(energies[1:,j]-energies[:-1,j])/dk, '.',
    #              color=color, markersize=1)
    #
    # ax.grid()
    # fig.tight_layout()
    # fig.savefig("img/mp_group_velocity.png")
    # # plt.show()
    # plt.close(fig)


def MPpresence():
    global N, R, Bprime, Bz

    num_arr = np.arange(2, 11)
    B0_arr = np.linspace(0.1, 2, num=50)

    s = np.zeros((num_arr.shape[0], B0_arr.shape[0]))
    for c1, num_sites in enumerate(num_arr):
        N = num_sites
        R = a * np.arange(0, N)
        Bprime = np.zeros((N, 3, 3*N))

        klim = [1e8, np.pi/(N*a)]
        k_arr = np.linspace(*klim, num=num_k)
        dk = k_arr[1] - k_arr[0]

        dim = 16*N
        energies = np.zeros((num_k,dim), dtype=complex)
        U = np.zeros((num_k,dim,dim), dtype=complex)
        z = np.zeros((num_k,dim))
        for j in range(N):
            Bprime[j,0,3*j] = 3.0  # d(Bx)/dx
            Bprime[j,1,3*j] = 0.0  # d(By)/dx
            Bprime[j,2,3*j] = 0.0  # d(Bz)/dx

            Bprime[j,0,3*j+1] = 0.0  # d(Bx)/dy
            Bprime[j,1,3*j+1] = 0.0  # d(By)/dy
            Bprime[j,2,3*j+1] = 0.0  # d(Bz)/dy

            Bprime[j,0,3*j+2] = 0.0  # d(Bx)/dz
            Bprime[j,1,3*j+2] = 0.0  # d(By)/dz
            Bprime[j,2,3*j+2] = 0.0  # d(Bz)/dz

        for c2, B0 in enumerate(B0_arr):
            Bz = B0 * np.ones(N)

            for i, k in enumerate(k_arr):
                bosonic_system = BosonicSystem(create_Tmatrix(k), create_Umatrix(k))
                energies[i,:], U[i,:,:] = bosonic_system.diagonalize()
                for j in range(dim):
                    z[i,j] = np.sum(np.abs(U[i,:N,j])**2) + np.sum(np.abs(U[i,4*N:5*N,j])**2) + \
                        np.sum(np.abs(U[i,8*N:9*N,j])**2) + np.sum(np.abs(U[i,12*N:13*N,j])**2)

            for j in range(dim):
                s[c1,c2] += simps(np.logical_and(0.4 < z[:,j], z[:,j] < 0.6), k_arr)
    np.save("data/NvsBz_const.npy", s)

    plt.imshow(np.transpose(np.log(s)), origin="bottom", aspect='auto',
               extent=(num_arr[0], num_arr[-1]+1, B0_arr[0], B0_arr[-1]))
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("img/NvsBz_const.png")
    plt.show()


def plot_MPpresence(filename):
    s = np.load("data/%s.npy" % filename)
    num_arr = np.arange(2, 11)
    B0_arr = np.linspace(0.1, 2, num=50)

    plt.imshow(np.transpose(s), origin="bottom", aspect='auto',
               extent=(num_arr[0], num_arr[-1]+1, B0_arr[0], B0_arr[-1]))
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("img/%s.png" % filename)
    plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Useful Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def create_Tmatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, c{-k,1...3*N}]
    Tmatrix = np.zeros((8*N, 8*N), dtype=complex)

    phi, eps_arr = order_eig(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)
    Dprime_k, Dprime_0 = Dprime_tensor(k), Dprime_tensor(0)

    # magnon terms
    for j1 in range(N):
        Tmatrix[j1,j1] += mu_B*g*S*Bz[j1]
        Tmatrix[4*N+j1,4*N+j1] += mu_B*g*S*Bz[j1]
        for j2 in range(N):
            Tmatrix[j1,j1] += 0.5*S * J0[j1,j2]
            Tmatrix[j2,j2] += 0.5*S * J0[j1,j2]
            Tmatrix[j1,j2] -= 0.5*np.sqrt(S*S) * \
                (Jk[j1,j2] + np.conjugate(Jk[j2,j1]))
            Tmatrix[j1,j2] -= 0.5j*np.sqrt(S*S) * \
                (Dk[j1,j2] - np.conjugate(Dk[j2,j1]))

            Tmatrix[4*N+j1,4*N+j1] += 0.5*S * J0[j1,j2]
            Tmatrix[4*N+j2,4*N+j2] += 0.5*S * J0[j1,j2]
            Tmatrix[4*N+j1,4*N+j2] -= 0.5*np.sqrt(S*S) * \
                (np.conjugate(Jk[j1,j2]) + Jk[j2,j1])
            Tmatrix[4*N+j1,4*N+j2] -= 0.5j*np.sqrt(S*S) * \
                (np.conjugate(Dk[j1,j2]) - Dk[j2,j1])

    # phonon terms
    for lambd in range(3*N):
        w = np.sqrt(phi[lambd] / m)
        Tmatrix[N+lambd,N+lambd] += hbar*w
        Tmatrix[5*N+lambd,5*N+lambd] += hbar*w

    # magnon-phonon interaction
    for j in range(N):
        for lambd in range(3*N):
            w = np.sqrt(phi[lambd] / m)
            Bterm = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                (Bprime[j,0,:]+1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )
            Bterm_c = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                (Bprime[j,0,:]-1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )

            Tmatrix[j,N+lambd] += Bterm
            Tmatrix[5*N+lambd,4*N+j] += Bterm_c

            Tmatrix[4*N+j,5*N+lambd] += np.conjugate(Bterm_c)
            Tmatrix[N+lambd,j] += np.conjugate(Bterm)

    return Tmatrix


def create_Umatrix(k):
    # alpha = [a{k,1...N}, c{k,1...3*N}, c{-k,1...3*N}]
    Umatrix = np.zeros((8*N, 8*N), dtype=complex)

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
            Bterm = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                (Bprime[j,0,:]+1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )
            Bterm_c = np.sum(-mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                (Bprime[j,0,:]-1j*Bprime[j,1,:])*eps_arr[:,lambd]
            )

            Umatrix[j,5*N+lambd] += Bterm
            Umatrix[5*N+lambd,j] += Bterm

            Umatrix[4*N+j,N+lambd] += np.conjugate(Bterm_c)
            Umatrix[N+lambd,4*N+j] += np.conjugate(Bterm_c)

    return Umatrix


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


def show_matrix(X):
    plt.imshow(X)
    plt.grid()
    plt.colorbar()
    plt.show()


def order_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(np.real(vals))[::-1]
    return vals[i_arr], vecs[:, i_arr]


def phonon_energy(k, j=0, linear_dispersion=False):
    if(linear_dispersion is True):
        return hbar * phonon_speeds[j] * k
    else:
        return 2*hbar*np.sqrt(Phi[j,j]/m) * np.abs(np.sin(0.5*k*a))


def magnon_energy(k, j=0):
    return 2*J*S*(1-np.cos(k*a)) - 2*Dz*S*np.sin(k*a) + mu_B*g*Bz[j]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
