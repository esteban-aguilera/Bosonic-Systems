# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from scipy.integrate import dblquad

import diagonalization_routines as diag
from utils import *


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# class
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class BosonicSystem:

    def __init__(self, createH, **kwargs):
        """Create a system composed of bosonic N bosonic operators.

        Params:
            T: Matrix that represent the dagger(a_i)*a_j terms
            U: Matrixs that represent the dagger(a_i)*dagger(a_j) terms
        """
        self.createH = createH

        Hamiltonian = createH(0)
        if((len(Hamiltonian.shape) != 2) or
           (Hamiltonian.shape[0] != Hamiltonian.shape[1]) or
           (Hamiltonian.shape[0] % 2 != 0)):
            raise ValueError('Hamiltonian\'s shape must be (2*N, 2*N), '
                'but it has shape {0}'.format(Hamiltonian.shape))
        self.ndim = Hamiltonian.shape[0]

        self.dmethod = kwargs.get('dmethod', 'Colpa').lower()

    def diagonalize(self, k):
        if(self.dmethod == 'bogoliubov'):
            return diag.bogoliubov( self.createH(k) )
        elif(self.dmethod == 'colpa'):
            return diag.colpa( self.createH(k) )

    def get_dispersion(self, k_arr, fn=None):
        num_k = k_arr.shape[0]

        energies = np.zeros((num_k, self.ndim), dtype=complex)
        U = np.zeros((num_k, self.ndim, self.ndim), dtype=complex)
        for i, k in enumerate(k_arr):
            energies[i,:], U[i,:,:] = self.diagonalize(k)

        if(fn is not None):
            np.save(fn, [energies, U])

        return energies, U

    def bcurvature(self, i, k, dk):
        N = self.ndim // 2
        J = np.diag( np.concatenate([np.ones(N), -np.ones(N)]) )

        def Pj(k, j):
            Gamma = np.zeros((2*N, 2*N))
            Gamma[j,j] = 1
            energies, T = self.diagonalize(k)
            return np.dot(T, np.dot(Gamma, np.dot(J, np.dot(HermitianConjugate(T), J))))

        P = Pj(k, i)
        dPx = (Pj([k[0]+dk,k[1],k[2]], i)-Pj([k[0]-dk,k[1],k[2]], i)) / (2*dk)
        dPy = (Pj([k[0],k[1]+dk,k[2]], i)-Pj([k[0],k[1]-dk,k[2]], i)) / (2*dk)

        Omega = np.trace(np.dot(np.identity(2*N)-P, np.dot(dPx, dPy))) - \
            np.trace(np.dot(np.identity(2*N)-P, np.dot(dPy, dPx)))

        return -np.imag(Omega)

    def chern_number(self, a, b, lower_limit, upper_limit, n=None, dk=1e-7):
        if(n is None):
            chern = np.zeros(self.ndim)
            for i in range(self.ndim):
                val, err = dblquad(lambda y, x: self.bcurvature(i, [x, y, 0], dk),
                                   a, b, lower_limit, upper_limit,
                                   epsabs=2*dk, epsrel=2*dk)
                chern[i] = val / (2*np.pi)

            return chern
        else:
            val, err = dblquad(lambda y, x: self.bcurvature(n, [x, y, 0], dk),
                               a, b, lower_limit, upper_limit,
                               epsabs=2*dk, epsrel=2*dk)

            return val / (2*np.pi)

    def plot_bcurvature(self, n, a, b, lim_inf, lim_sup, num=1000, fn=None,
                        logscale=False, show=True):
        kx_arr = np.linspace(a, b, num=num)
        dk = kx_arr[1] - kx_arr[0]
        ky_arr = np.array(
            [np.arange(lim_inf(kx), lim_sup(kx)+dk, dk) for kx in kx_arr]
        )

        Omega_arr = np.array(
            [[self.bcurvature(n, [kx, ky, 0], dk) for ky in ky_arr[i]] for i, kx in enumerate(kx_arr)]
        )

        Kx = np.array(
            [kx_arr[i] for i in range(len(kx_arr)) for j in range(len(ky_arr[i]))]
        )
        Ky = np.array(
            [ky_arr[i][j] for i in range(len(kx_arr)) for j in range(len(ky_arr[i]))]
        )
        Bc = np.array(
            [Omega_arr[i][j] for i in range(len(kx_arr)) for j in range(len(ky_arr[i]))]
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if(logscale is True):
            sc = ax.scatter(Kx, Ky, c=np.log(np.abs(Bc)))
        else:
            sc = ax.scatter(Kx, Ky, c=Bc)

        fig.colorbar(sc, ax=ax)

        fig.tight_layout()
        if(fn is not None):
            fig.savefig(fn)
        if(show is True):
            plt.show()

        plt.close(fig)


    def plot_dispersion(self, k_arr, eigs=False, p_arr=None,
        cbar_label=None, track=None, xlim=None, ylim=None, y_arr=None,
        xticks=[], xlabels=[], yticks=[], ylabels=[], title='', fn=None,
        show=True, ret=False):
        if(eigs is True):
            energies, U = k_arr
        else:
            energies, U = self.get_dispersion(k_arr)

        # create plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if(p_arr is None):
            for i in range(self.ndim // 2):
                if(self.dmethod == 'bogoliubov'):
                    ax.plot(np.imag(energies[:,i]), color='red')
                ax.plot(np.real(energies[:,i]), color='darkblue')
        else:
            z = np.zeros((U.shape[0], self.ndim))
            for i, k in enumerate(k_arr):
                for j in range(self.ndim):
                    if(track == 'angle'):
                        z[i,j] = np.angle(U[i,p_arr,j])
                    else:
                        z[i,j] = np.sum(np.abs(U[i,p_arr,j])**2)

            for j in range(self.ndim):
                x, y = np.arange(energies.shape[0]), np.real(energies[:,j])

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = plt.Normalize(vmin=0, vmax=1)
                lc = LineCollection(segments, cmap='brg', norm=norm)
                lc.set_array(z[:,j])
                line = ax.add_collection(lc)
            cbar = fig.colorbar(line, ax=ax, ticks=[0.0, 1.0])
            if(cbar_label is not None):
                cbar.ax.set_yticklabels(cbar_label)

        if(y_arr is not None):
            x_arr = np.arange(y_arr.shape[1])
            for i in range(y_arr.shape[0]):
                ax.plot(x_arr, y_arr[i,:], color='black')

        ax.set_title(title)

        if(xlim is None):
            ax.set_xlim([0, energies.shape[0]])
        else:
            ax.set_xlim(xlim)

        if(ylim is None):
            ax.set_ylim([0.95*np.min(np.abs(energies)), 1.05*np.max(np.abs(energies))])
        else:
            ax.set_ylim(ylim)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=12)
        ax.xaxis.grid(True)

        if(yticks != [] and ylabels != []):
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=12)
            ax.yaxis.grid(True)

        fig.tight_layout()
        if(fn is not None):
            fig.savefig(fn)

        if(show is True):
            plt.show()

        if(ret is True):
            return fig, ax
        else:
            plt.close(fig)
