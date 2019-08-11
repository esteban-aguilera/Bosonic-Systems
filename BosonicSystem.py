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

        self.J = np.diag(np.concatenate((np.ones(self.N), -np.ones(self.N))))

    def diagonalize(self):
        energies, U = np.linalg.eig(np.dot(self.J, self.H))
        i_arr = np.argsort(-energies)
        return energies[i_arr], U[:,i_arr]

    def energies(self):
        energies = np.linalg.eigvals(np.dot(self.J, self.H))
        i_arr = np.argsort(-energies)
        return np.real(energies[i_arr][:self.N])

    def plot_H(self):
        plt.imshow(np.real(self.H))
        plt.colorbar()
        plt.grid()
        plt.show()

    def plot_eigvalsH(X):
        eigvals, _ = np.linalg.eigh(self.H)
        plt.plot(np.arange(eigvals.shape[0]), eigvals, '.')
        plt.grid()
        plt.show()


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # properties
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def x(self):
        return self._x

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # setter
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    @x.setter
    def x(self, value):
        self._x = C * value


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    my_class = MyClass(5)
    print(my_class.x)
