import numpy as np


def HermitianConjugate(X):
    return np.conjugate(np.transpose(X))


def ordered_eig(X):
    evals, evecs = np.linalg.eig(X)
    i_arr = np.argsort(-np.real(evals))
    return evals[i_arr], evecs[:, i_arr]


def paraunitary_eig(X):
    N = X.shape[0]//2
    evals, evecs = np.linalg.eig(X)
    i_arr = np.argsort(-np.real(evals))
    j_arr = np.concatenate([i_arr[:N][::-1],i_arr[N:]])
    return evals[j_arr], evecs[:,j_arr]


def karr_path(*args, num=1000, get_nums=False):
    if(len(args) == 0):
        return None
    elif(len(args) == 1):
        return args[0]
    else:
        kfraction = [np.sqrt(np.sum((args[i+1]-args[i])**2)) for i in range(len(args)-1)]
        ktotal = np.sum(kfraction)

        nums = [int(num*kfraction[n]/ktotal+0.5) for n in range(len(args)-1)]
        segments = [create_segment(args[i], args[i+1], num=nums[i]+(i!=len(args)-1))
                    for i in range(len(args)-1)]

        karr = segments[0][:-1]
        for i in range(1, len(segments)-1):
            karr = np.concatenate((karr, segments[i][:-1]), axis=0)
        karr = np.concatenate((karr, segments[-1]), axis=0)

        if(get_nums is True):
            return karr, nums
        else:
            return karr


def karr_circle(R, num=1000):
    phi = 2*np.pi/(num-1) * np.arange(num)
    karr = np.zeros((num,3))
    for i in range(num):
        karr[i,0] = R * np.cos(phi[i])
        karr[i,1] = R * np.sin(phi[i])
        karr[i,2] = 0

    return karr


def create_segment(x0, x1, num=300):
    x_arr = [[(1-i/(num-1))*x0[k]+i/(num-1)*x1[k] for k in range(3)] for i in range(num)]
    return np.array(x_arr)
