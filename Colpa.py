# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.integrate import simps


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
N = 1
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
Bprime = np.zeros((N, 3, 3*N)) + 1.0
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    # dispersion_relation(np.linspace(0.88e9, 1.05e9, num=1000), ylim=[4.0,4.6])
    dispersion_relation(np.linspace(0.3e9, .7e9, num=1000), ylim=[0, 3])

    # H = createH(0.3e9)
    # show_matrix(np.imag(H))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# class
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class Colpa:

    def __init__(self, Hamiltonian):
        self.H = Hamiltonian
        self.N = Hamiltonian.shape[0] // 2

    def diagonalize(self):
        J = np.diag(np.concatenate((np.ones(self.N), -np.ones(self.N))))

        ch = np.transpose(np.conjugate( np.linalg.cholesky(self.H) ))
        evals, U = ordered_eig( np.dot(ch,np.dot(J, T(ch))) )

        E = np.dot(J, np.diag(evals))
        Tbog = np.dot(np.linalg.inv(ch), np.dot(U, np.sqrt(E)))

        return E, Tbog


    def imshow(self, option=0):
        if(option == 0):
            plt.imshow(np.abs(self.H))
        elif(option == 1):
            plt.imshow(np.real(self.H))
        elif(option == 2):
            plt.imshow(np.imag(self.H))
        elif(option == 3):
            plt.imshow(np.imag(self.H + np.transpose(self.H)))

        plt.colorbar()
        plt.tight_layout()
        plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def dispersion_relation(k_arr, ylim=None):
    dim = 8*N

    num_k = k_arr.shape[0]
    dk = k_arr[1] - k_arr[0]
    klim = np.array([k_arr[0], k_arr[-1]])

    energies = np.zeros((num_k,dim), dtype=complex)
    U = np.zeros((num_k,dim,dim), dtype=complex)
    z = np.zeros((num_k,dim))

    for i, k in enumerate(k_arr):
        colpa = Colpa(createH(k))
        E, U[i,:,:] = colpa.diagonalize()
        energies[i,:] = np.real(np.diag(E))
        for j in range(dim):
            z[i,j] = np.sum(np.abs(U[i,:N,j])**2) + np.sum(np.abs(U[i,4*N:5*N,j])**2)

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
    if(ylim is None):
        ax.set_ylim(np.real(energies).min(), np.real(energies).max())
        ax.text(0.97*klim[1], 0.97*np.real(energies).max(), "$B_z=%.3f$" % Bz[0],
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    else:
        ax.set_ylim(ylim)
        ax.text(0.97*klim[1], 0.97*ylim[1], "$B_z=%.3f$" % Bz[0],
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    fig.tight_layout()
    fig.savefig("img/colpa_dispersion.png")
    plt.show()
    plt.close(fig)


def createH(k, N=1):
    H = np.zeros((8*N, 8*N), dtype=complex)

    phi, eps_arr = ordered_eig(Phi_matrix(k))
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


def ordered_eig(X):
    vals, vecs = np.linalg.eigh(X)
    i_arr = np.argsort(-np.real(vals))
    return vals[i_arr], vecs[:, i_arr]


def show_matrix(X):
    plt.imshow(X)
    plt.grid()
    plt.colorbar()
    plt.show()


def T(X):
    return np.conjugate(np.transpose(X))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
