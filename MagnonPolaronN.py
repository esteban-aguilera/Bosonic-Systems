# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.integrate import simps

from Colpa import *
from generate_gif import *


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
Nsites = 1  # Number of sites
a = 12.376e-10  # interatomic distance

# phonon constants
hbar = 6.582e-13
m = 9.8e-24

phonon_speeds = np.array([3843, 3843, 7209])
Phi = m * np.diag(phonon_speeds/a)**2

# magnon constants
R = a * np.arange(0, Nsites)
mu_B = 5.788e-2
g = 2
S = 20
J = 0.17
Dz = 0
Bz = np.zeros(Nsites)
Bz = 0.0 * np.ones(Nsites)  # 1.30

# magnon polarons constants
Bprime = np.zeros((Nsites, 3, 3*Nsites)) + 1
for j in range(Nsites):
    Bprime[j,0,3*j] += 0.0  # 2*np.pi/(N*a)*np.cos(2*np.pi/(N*a)*R[j])  # d(Bx)/dx
    Bprime[j,1,3*j] += 0.0  # d(By)/dx
    Bprime[j,2,3*j] += 0.0  # d(Bz)/dx

    Bprime[j,0,3*j+1] += 0.0 # d(Bx)/dy
    Bprime[j,1,3*j+1] += 0.0  # d(By)/dy
    Bprime[j,2,3*j+1] += 0.0  # d(Bz)/dy

    Bprime[j,0,3*j+2] += 0.0  # d(Bx)/dz
    Bprime[j,1,3*j+2] += 0.0  # d(By)/dz
    Bprime[j,2,3*j+2] += 0.0  # d(Bz)/dz


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    # plot_dispersion(np.linspace(1e7, np.pi/(Nsites*a), num=1000), show_plot=True)
    # plot_dispersion(np.linspace(0.3e9, np.pi/(Nsites*a), num=1000), show_plot=True)
    plot_dispersion(np.linspace(0.3e9, 0.7e9, num=1000), show_plot=True)

    # plot_evolution(0.5e9)

    # video_sound_speed(np.linspace(2000, 7500, num=100))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def createH(k, N=Nsites, a=a, phonon_speeds=phonon_speeds, J=J, Dz=Dz, Bz=Bz,
            Bprime=Bprime, S=S, m=m, hbar=hbar, mu_B=mu_B, g=g):
    R = a * np.arange(0, N)
    # Phi = m * np.diag(phonon_speeds/a)**2

    # useful functions
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

    H = np.zeros((8*N, 8*N), dtype=complex)
    phi, eps_arr = ordered_eigh(Phi_matrix(k))
    Jk, J0 = J_matrix(k), J_matrix(0)
    Dk, D0 = Dz_matrix(k), Dz_matrix(0)

    # magnon term
    for j1 in range(N):
        H[j1,j1] += mu_B*g*S*Bz[j1]
        H[4*N+j1,4*N+j1] += mu_B*g*S*Bz[j1]
        for j2 in range(N):
            H[j1,j1] += 0.5*S * J0[j1,j2]
            H[j2,j2] += 0.5*S * J0[j1,j2]
            H[j1,j2] -= 0.5*S * (Jk[j1,j2] + np.conjugate(Jk[j2,j1]))
            H[j1,j2] -= 0.5j*S * (Dk[j1,j2] - np.conjugate(Dk[j2,j1]))

            H[4*N+j1,4*N+j1] += 0.5*S * J0[j1,j2]
            H[4*N+j2,4*N+j2] += 0.5*S * J0[j1,j2]
            H[4*N+j2,4*N+j1] -= 0.5*S * (np.conjugate(Jk[j1,j2]) + Jk[j2,j1])
            H[4*N+j2,4*N+j1] -= 0.5j*S * (np.conjugate(Dk[j1,j2]) - Dk[j2,j1])

    # phonon term
    for lambd in range(3*N):
        w = np.sqrt(phi[lambd] / m)
        H[N+lambd,N+lambd] += hbar*w
        H[5*N+lambd,5*N+lambd] += hbar*w

    # magnon-phonon coupling
    for j in range(N):
        for lambd in range(3*N):
            w = np.sqrt(phi[lambd] / m)
            Bterm = -mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                np.dot(Bprime[j,0,:]+1j*Bprime[j,1,:], eps_arr[:,lambd])
            Bterm_c = -mu_B*g*np.sqrt(hbar*S/(4*m*w)) * \
                np.dot(Bprime[j,0,:]-1j*Bprime[j,1,:], eps_arr[:,lambd])

            H[j,N+lambd] += Bterm
            H[j,5*N+lambd] += Bterm
            H[4*N+j,N+lambd] += Bterm_c
            H[4*N+j,5*N+lambd] += Bterm_c

            H[5*N+lambd,4*N+j] += np.conjugate(Bterm_c)
            H[N+lambd,4*N+j] += np.conjugate(Bterm_c)
            H[5*N+lambd,j] += np.conjugate(Bterm)
            H[N+lambd,j] += np.conjugate(Bterm)

    return H


def plot_dispersion(k_arr, N=Nsites, ylim=None,
                    path="img", fn="MagnonPolaronN", show_plot=False):
    dim = 8*N
    num_k = k_arr.shape[0]
    dk = k_arr[1] - k_arr[0]
    klim = [k_arr[0], k_arr[-1]]

    z = np.zeros((num_k, dim))

    colpa = Colpa(createH, dim//2)
    energies, U = colpa.dispersion_relation(k_arr)
    for i, k in enumerate(k_arr):
        for j in range(dim):
            z[i,j] = np.sum(np.abs(U[i,:N,j])**2) + np.sum(np.abs(U[i,4*N:5*N,j])**2)

    if(ylim is None):
        ylim = [0, np.real(energies).max()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(dim//2):
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
    ax.set_ylim(ylim)
    ax.text(0.97*klim[1], 0.97*ylim[1], "$B_z=%.3f$" % Bz[0],
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    fig.tight_layout()
    fig.savefig("%s/%s.png" % (path, fn))
    if(show_plot is True):
        plt.show()

    plt.close(fig)

    return energies, U


def plot_evolution(k, dt=1e-14, tf=1e-10, N=Nsites, hbar=hbar):
    t_arr = np.arange(0, tf, dt)
    dim = 8*N

    Phi_t = np.zeros((dim, t_arr.shape[0]+1), dtype=complex)
    Psi0 = np.zeros(dim, dtype=complex)
    Psi0[2] = 1

    colpa = Colpa(createH, dim//2)
    energies, U = colpa.diagonalize(k)
    for i, t in enumerate(t_arr):
        Phi_t[:,i] = np.dot(U, np.exp(-1j*energies*i*dt/hbar)*np.dot(T(U), Psi0) )

    plt.imshow(np.abs(Phi_t)**2, aspect='auto')
    plt.colorbar()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return Phi_t


def video_sound_speed(speeds):
    global Phi

    filenames = [None for i in range(speeds.shape[0])]
    for i in range(speeds.shape[0]):
        phonon_speeds = np.array([3843, speeds[i], 7209])
        Phi = m * np.diag(phonon_speeds/a)**2
        fn = "MagnonPolaronN-%d" % (i+1)
        plot_dispersion(np.linspace(0.4e9, 1.1e9, num=1000), fn=fn)
        filenames[i] = fn
    generate_gif(filenames, "MagnonPolaronSpeeds")


def ordered_eigh(X):
    vals, vecs = np.linalg.eigh(X)
    i_arr = np.argsort(-np.real(vals))
    return vals[i_arr], vecs[:, i_arr]


def ordered_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(-np.real(vals))
    return vals[i_arr], vecs[:, i_arr]


def T(X):
    return np.transpose(np.conjugate(X))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
