import numpy as np

def samples_generation(predictor_plant, objective, nx):
    n   = 2 * nx + 1
    X   = np.random.rand(10*n,nx)
    f   = np.zeros([10*n,nx])
    min = np.inf
    p   = 0
    for i in range(10*n):
        k = 0
        for j in range(np.shape(predictor_plant)[0]):
            if predictor_plant[j](X[i])>=0 and  abs(objective(X[i]))<0.13:
                k     += 1
        if k==np.shape(predictor_plant)[0]:
            p      += 1
            f[p-1, :] = X[i]
            if min > objective(X[i]):
                min  = objective(X[i])
                mink = p-1
            if p>=n:break

    return f[:p], mink