import numpy as np


def HermitianConjugate(X):
    return np.conjugate(np.transpose(X))


def paraunitary_eig(X):
    N = X.shape[0]//2
    evals, evecs = np.linalg.eig(X)
    i_arr = np.argsort(-np.real(evals))
    j_arr = np.concatenate([i_arr[:N],i_arr[N:][::-1]])
    return evals[j_arr], evecs[:, j_arr]


def create_karr(*args, nums=None):
    if(len(args) == 0):
        return None
    elif(len(args) == 1):
        return args[0]
    else:
        if(nums is None):
            data = [create_line(args[i], args[i+1]) for i in range(len(args)-1)]
        elif(len(nums) == len(args)-1):
            data = [create_line(args[i], args[i+1], num=nums[i]) for i in range(len(args)-1)]

        out = data[0]
        for i in range(1, len(data)):
            out = np.concatenate((out, data[i]), axis=0)

        return out


def create_line(x0, x1, num=300):
    x_arr = [[(1-i/(num-1))*x0[k]+i/(num-1)*x1[k] for k in range(3)] for i in range(num)]
    return np.array(x_arr)
