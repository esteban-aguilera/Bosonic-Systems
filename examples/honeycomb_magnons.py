import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')

from BosonicSystem import BosonicSystem
from utils import create_karr


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
muB = 5.788e-2  # Bohr's magneton
g = 2  # g-factor

S = 1.5  # spin value
J = 1  # Heisenberg exchange
Dz = 0  # z-component of the Dzyaloshinskii-Moriya interaction.
Bz = 4  # z-component of the magnetic field


a = 1  # nearest neighbor distance
delta_arr = np.array(
    [[a, 0, 0],
     [-0.5*a, 0.5*np.sqrt(3)*a, 0],
     [-0.5*a, -0.5*np.sqrt(3)*a, 0]]
)

# symmetry points
Gamma = np.array( [0, 0, 0] )
M = 2*np.pi/(3*a) * np.array( [1, 0, 0] )
Kp = 2*np.pi/(3*a) * np.array( [1, 1/np.sqrt(3), 0] )
Kn = 2*np.pi/(3*a) * np.array( [1, -1/np.sqrt(3), 0] )

# k-space path
kpath = Gamma, M, Kp, Gamma
numk = 10000


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def main():
    for dmethod in ['Bogoliubov', 'Colpa']:
        k_arr, nums = create_karr(*kpath, num=numk, get_nums=True)

        xticks = [np.sum(nums[:n]) for n in range(len(nums)+1)]
        xlabels = ['$\Gamma$', '$M$', '$K^+$', '$\Gamma$']

        bs = BosonicSystem(createH)
        bs.plot_dispersion(k_arr, title='Honeycomb Magnons with %s\'s Method' % dmethod,
                           xticks=xticks, xlabels=xlabels, show=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def createH(k):
    if((type(k) == float or type(k) == int) and k == 0):
        k = np.array([0, 0, 0])

    if(len(np.array(k).shape) > 1 or np.array(k).shape[0] != 3):
        raise ValueError('k vector must be three-dimensional')

    fp = np.sum([np.exp(1j*np.dot(k[:], delta_arr[i,:])) for i in range(3)])
    fn = np.sum([np.exp(-1j*np.dot(k[:], delta_arr[i,:])) for i in range(3)])

    return np.array(
        [[g*Bz-S*J,             0.5*S*(2*J-1j*Dz)*fp, 0,                    0                   ],
         [0.5*S*(2*J+1j*Dz)*fn, g*Bz-S*J,             0,                    0                   ],
         [0,                    0,                    g*Bz-S*J,             0.5*S*(2*J+1j*Dz)*fp],
         [0,                    0,                    0.5*S*(2*J-1j*Dz)*fn, g*Bz-S*J            ]]
    )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# main
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
