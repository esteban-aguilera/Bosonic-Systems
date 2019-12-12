import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')

from BosonicSystem import BosonicSystem
from utils import create_karr


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
hbar = 6.582e-13  # Planck reduced constant [meV*s]
muB = 5.788e-2  # Bohr's magneton [meV]
g = 2  # g-factor

a = 12.376e-10  # nearest neighbor distance

m = 9.8e-24  # mass
omega0 = 7209 / a  # natural frequency of phonons.

S = 20  # spin value
J = 0.17  # Heisenberg exchange
Dz = 0  # z-component of the Dzyaloshinskii-Moriya interaction.
Bz = 1  # z-component of the magnetic field

Bperp = 8.24e-9


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    for dmethod in ['Bogoliubov', 'Colpa']:
        bs = BosonicSystem(createH)

        k_arr = np.linspace(-np.pi/a, np.pi/a, num=1000)

        bs.plot_dispersion(k_arr, p_arr=[0,2],
            title='Magnon-Phonon Interaction with %s\'s Method' % dmethod, show=True
        )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def createH(k):
    if((type(k) not in [int, np.int32, float, np.float64])):
        raise ValueError('k must be a real number')
    elif(k == 0):
        return np.zeros((4, 4))

    Em = 2*S*J*(1-np.cos(k*a)) - 2*S*Dz*np.sin(k*a) + muB*g*Bz
    Ep = hbar*omega0*abs(np.sin(k*a/2))
    Gamma = 1 * Bperp*np.sqrt(hbar**2/(4*m*S*Ep)) * k

    return np.array(
        [[Em,     Gamma, 0,      0    ],
         [Gamma, Ep,    0,      0    ],
         [0,      0,     Em,     Gamma],
         [0,      0,     Gamma, Ep   ]]
    )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
