# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# class
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class BosonicSystem:

    def __init__(self, T, U):
        """Create a system composed of bosonic N bosonic operators.

        Params:
            T: Matrix that represent the dagger(a_i)*a_j terms
            U: Matrixs that represent the dagger(a_i)*dagger(a_j) terms
        """
        assert(T.shape[0] == T.shape[1])
        assert(U.shape[0] == U.shape[1])
        assert(T.shape[0] == U.shape[0])
        self.N = T.shape[0]

        # Create full Hamiltonian matrix
        A = np.concatenate((T, U), axis=1)
        B = np.concatenate((np.conjugate(U), np.conjugate(T)), axis=1)
        self.H = np.concatenate((A, B), axis=0)

        vals = np.linalg.eigvalsh(self.H)
        if(np.any(vals < -1e-7)):
            print("Hamiltonian is not positive definite!", np.min(vals))

        self.J = np.diag(np.concatenate((np.ones(self.N), -np.ones(self.N))))

    def diagonalize(self):
        paraH = np.dot(self.J, self.H)
        energies, U = np.linalg.eig(paraH)
        i_arr = np.argsort(np.real(energies))[::-1]
        return energies[i_arr], U[:,i_arr]

    def plot_H(self):
        plt.imshow(np.log(np.abs(self.H)))
        plt.colorbar()
        plt.grid()
        plt.show()

    def plot_eigvalsH(self):
        eigvals, _ = np.linalg.eigh(self.H)
        plt.plot(np.arange(eigvals.shape[0]), eigvals, '.')
        plt.grid()
        plt.show()
