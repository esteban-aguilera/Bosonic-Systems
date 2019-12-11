# coding: utf-8
# !/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from utils import *


def bogoliubov(Hamiltonian):
    if((len(Hamiltonian.shape) != 2) or
       (Hamiltonian.shape[0] != Hamiltonian.shape[1]) or
       (Hamiltonian.shape[0] % 2 != 0)):
        raise ValueError('Hamiltonian\'s shape must be (2*N, 2*N), but it has shape {0}'.format(Hamiltonian.shape))

    N = Hamiltonian.shape[0] // 2
    J = np.diag(np.concatenate([np.ones(N), -np.ones(N)]))

    paraH = np.dot(J, Hamiltonian)
    energies, T = paraunitary_eig( paraH )

    return np.concatenate([energies[:N], -energies[N:]]), T


def colpa(Hamiltonian):
    if((len(Hamiltonian.shape) != 2) or
       (Hamiltonian.shape[0] != Hamiltonian.shape[1]) or
       (Hamiltonian.shape[0] % 2 != 0)):
        raise ValueError('Hamiltonian\'s shape must be (2*N, 2*N), but it has shape {0}'.format(Hamiltonian.shape))

    N = Hamiltonian.shape[0] // 2
    J = np.diag(np.concatenate([np.ones(N), -np.ones(N)]))
    try:
        H = HermitianConjugate( np.linalg.cholesky(Hamiltonian) )
        L, U = paraunitary_eig( np.dot(np.dot(H,J), HermitianConjugate(H)) )

        E = np.dot(J, np.diag(L))
        T = np.linalg.inv( np.dot(np.linalg.inv(H), np.dot(U, np.sqrt(E))) )
    except np.linalg.LinAlgError:  # Hamiltonian is not definite-positive.
        E = -np.ones((2*N, 2*N), dtype=complex)
        T = np.zeros((2*N, 2*N), dtype=complex)

    return np.real(np.diag(E)), T
