# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import diagonalization_routines as diag


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

        self.dmethod = kwargs.get('dmethod', 'bogoliubov').lower()

    def diagonalize(self, k):
        if(self.dmethod == 'bogoliubov'):
            return diag.bogoliubov( self.createH(k) )
        elif(self.dmethod == 'colpa'):
            return diag.colpa( self.createH(k) )

    def get_dispersion(self, k_arr, fn=None):
        num_k = k_arr.shape[0]

        energies = np.zeros((num_k, self.ndim))
        U = np.zeros((num_k, self.ndim, self.ndim), dtype=complex)
        for i, k in enumerate(k_arr):
            energies[i,:], U[i,:,:] = self.diagonalize(k)

        if(fn is not None):
            np.save(fn, [energies, U])

        return energies, U

    def plot_dispersion(self, k_arr, xlim=None, ylim=None, title='',
                        xticks=[], xlabels=[], yticks=[], ylabels=[],
                        fn=None, show=True):
        # get dispersion relation
        energies, U = self.get_dispersion(k_arr)

        # create plot
        fig = plt.figure()
        ax = fig.add_subplot()

        for i in range(energies.shape[1]):
            ax.plot(np.real(energies[:,i]))

        ax.set_title(title)

        if(xlim is None):
            ax.set_xlim([0, energies.shape[0]])
        else:
            ax.set_xlim(xlim)

        if(ylim is None):
            ax.set_ylim([0, 1.05*np.max(energies)])
        else:
            ax.sett_ylim(ylim)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=12)
        ax.xaxis.grid(True)

        if(yticks != [] and ylabels != []):
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=12)
            ax.yaxis.grid(True)

        if(fn is not None):
            fig.savefig(fn)

        if(show is True):
            plt.show()
        plt.close(fig)
