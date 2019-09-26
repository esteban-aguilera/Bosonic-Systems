# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# class
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class Colpa:

    def __init__(self, createH, N):
        self.createH = createH
        self.N = N

        self.J = np.diag(np.concatenate([np.ones(self.N), -np.ones(self.N)]))

    def dispersion_relation(self, k_arr, savefile=False,
                            fn="colpa_dispersion", path="data"):
        num_k = k_arr.shape[0]
        dk = k_arr[1] - k_arr[0]
        klim = np.array([k_arr[0], k_arr[-1]])

        energies = np.zeros((num_k, 2*self.N), dtype=complex)
        U = np.zeros((num_k, 2*self.N, 2*self.N), dtype=complex)
        for i, k in enumerate(k_arr):
            energies[i,:], U[i,:,:] = self.diagonalize(k)

        if(savefile is True):
            np.save("%s/%s.npy" % (path, fn), [energies, U])

        return energies, U

    def diagonalize(self, k):
        Hk = self.createH(k)
        L = np.linalg.cholesky(Hk)
        evals, evecs = paraunitary_eigh( np.dot(np.dot(T(L),self.J), L) )

        E = np.dot(self.J, np.diag(evals))
        U = np.dot(np.linalg.inv(T(L)), np.dot(evecs, np.sqrt(E)))

        return np.diag(E), U


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def paraunitary_eigh(X):
    N = X.shape[0]//2
    vals, vecs = np.linalg.eigh(X)
    i_arr = np.argsort(-np.real(vals))
    j_arr = np.concatenate([i_arr[:N],i_arr[N:][::-1]])
    return vals[j_arr], vecs[:, j_arr]


def ordered_eigh(X):
    vals, vecs = np.linalg.eigh(X)
    i_arr = np.argsort(-np.real(vals))
    return vals[i_arr], vecs[:, i_arr]


def ordered_eig(X):
    vals, vecs = np.linalg.eig(X)
    i_arr = np.argsort(np.real(vals))
    return vals[i_arr], vecs[:, i_arr]


def T(X):
    return np.conjugate(np.transpose(X))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
