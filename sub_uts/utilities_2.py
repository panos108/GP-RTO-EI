# v2 includes shaping the TR with the curvature of the problem by a broyden update on derivatives
# and a BFGS update on the Hessian, however the TR becomes very small in some parts, so the approach
# does not seem to be too effective.

import time
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist

import sobol_seq
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy import linalg
import scipy
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse
from scipy.stats import norm
from casadi import *


class GP_model:

    ###########################
    # --- initializing GP --- #
    ###########################
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True, noise=None, GP_casadi = False):

        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim = X.shape[0], X.shape[1]
        self.ny_dim = Y.shape[1]
        self.multi_hyper = multi_hyper
        self.var_out = var_out
        self.GP_casadi = GP_casadi
        # normalize data
        self.X_mean, self.X_std = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm = (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

        # determine hyperparameters
        self.hypopt, self.invKopt = self.determine_hyperparameters(noise)
        self.meanfcn, self.varfcn = self.GP_predictor()
        #############################

    # --- Covariance Matrix --- #
    #############################

    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''
        dist = cdist(X_norm, X_norm, 'seuclidean', V=W) ** 2
        r = np.sqrt(dist)
        if kernel == 'RBF':
            cov_matrix = sf2 * np.exp(-0.5 * dist)
            return cov_matrix
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        elif kernel == 'Matern32':
            cov_matrix = sf2 * (1 + 3**0.5 * r)*np.exp(-r*3**0.5)
            return cov_matrix
        elif kernel == 'Matern52':
            cov_matrix = sf2 * (1 + 5**0.5 * r + 5/3 * r**2) * np.exp(-r*5**0.5)
            return cov_matrix
        else:
            print('ERROR no kernel with name ', kernel)

    ################################
    # --- Covariance of sample --- #
    ################################

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1, nx_dim), 'seuclidean', V=ell) ** 2
        cov_matrix = sf2 * np.exp(-.5 * dist)

        return cov_matrix

        ###################################

    # --- negative log likelihood --- #
    ###################################

    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        '''
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel = self.kernel

        W = np.exp(2 * hyper[:nx_dim])  # W <=> 1/lambda
        sf2 = np.exp(2 * hyper[nx_dim])  # variance of the signal
        sn2 = np.exp(2 * hyper[nx_dim + 1])  # variance of noise

        K = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K = K + (sn2 + 1e-8) * np.eye(n_point)  # (nxn) covariance matrix
        K = (K + K.T) * 0.5  # ensure K is simetric
        L = np.linalg.cholesky(K)  # do a cholesky decomposition
        logdetK = 2 * np.sum(
            np.log(np.diag(L)))  # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY = np.linalg.solve(L, Y)  # obtain L^{-1}*Y
        alpha = np.linalg.solve(L.T, invLY)  # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL = np.dot(Y.T, alpha) + logdetK  # construct the NLL

        return NLL

    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################

    def determine_hyperparameters(self, noise=None):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''
        # internal parameters
        X_norm, Y_norm = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim = self.kernel, self.ny_dim
        Cov_mat = self.Cov_mat

        lb = np.array([-5.] * (nx_dim + 1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub = np.array([5.] * (nx_dim + 1) + [-4.])  # lb on parameters (this is inside the exponential)



        bounds = np.hstack((lb.reshape(nx_dim + 2, 1),
                            ub.reshape(nx_dim + 2, 1)))
        multi_start = self.multi_hyper  # multistart on hyperparameter optimization
        multi_startvec = sobol_seq.i4_sobol_generate(nx_dim + 2, multi_start)

        options = {'disp': False, 'maxiter': 10000}  # solver options
        hypopt = np.zeros((nx_dim + 2, ny_dim))  # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.] * multi_start  # values for multistart
        localval = np.zeros((multi_start))  # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):
            if not(np.array([noise]).reshape((-1,))[i]==None):
                noise = np.array([noise]).reshape((-1,))
                lb[-1] = np.log(noise[i]/self.Y_std[i]**2)*0.5-0.001
                ub[-1] = np.log(noise[i]/self.Y_std[i]**2)*0.5+0.001
                bounds = np.hstack((lb.reshape(nx_dim + 2, 1),
                                    ub.reshape(nx_dim + 2, 1)))
            # --- multistart loop --- #
            for j in range(multi_start):
                # print('multi_start hyper parameter optimization iteration = ',j,'  input = ',i)
                hyp_init = lb + (ub - lb) * multi_startvec[j, :]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood, hyp_init, args=(X_norm, Y_norm[:, i]) \
                               , method='SLSQP', options=options, bounds=bounds, tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex = np.argmin(localval)
            hypopt[:, i] = localsol[minindex]
            ellopt = np.exp(2. * hypopt[:nx_dim, i])
            sf2opt = np.exp(2. * hypopt[nx_dim, i])
            sn2opt = np.exp(2. * hypopt[nx_dim + 1, i]) #+ 1e-8

            # --- constructing optimal K --- #
            Kopt = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt * np.eye(n_point)
            # --- inverting K --- #
            invKopt += [np.linalg.solve(Kopt, np.eye(n_point))]

        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################
    def GP_predictor(self):# , X,Y):
            nd, invKopt, hypopt = self.ny_dim, self.invKopt, self.hypopt
            Ynorm, Xnorm = SX(DM(self.Y_norm)), SX(DM(self.X_norm))
            ndat = Xnorm.shape[0]
            nX, covSEfcn = self.nx_dim, self.covSEard()
            stdX, stdY, meanX, meanY = SX(self.X_std), SX(self.Y_std), SX(self.X_mean), SX(self.Y_mean)
            #        nk     = 12
            x = SX.sym('x', nX)
            # nk     = X.shape[0]
            xnorm = (x - meanX) / stdX
            # Xnorm2 = (X - meanX)/stdX
            # Ynorm2 = (Y - meanY)/stdY

            k = SX.zeros(ndat)
            # k2     = SX.zeros(ndat+nk)
            mean = SX.zeros(nd)
            mean2 = SX.zeros(nd)
            var = SX.zeros(nd)
            # Xnorm2 = SX.sym('Xnorm2',ndat+nk,nX)
            # invKY2 = SX.sym('invKY2',ndat+nk,nd)

            for i in range(nd):
                invK = SX(DM(invKopt[i]))
                hyper = SX(DM(hypopt[:, i]))
                ellopt, sf2opt = exp(2 * hyper[:nX]), exp(2 * hyper[nX])
                for j in range(ndat):
                    k[j] = covSEfcn(xnorm, Xnorm[j, :], ellopt, sf2opt)
                # for j in range(ndat+nk):
                #    k2[j] = covSEfcn(xnorm,Xnorm2[j,:],ellopt,sf2opt)

                invKYnorm = mtimes(invK, Ynorm[:, i])
                mean[i] = mtimes(k.T, invKYnorm)
                # mean2[i]  = mtimes(k2.T,invKY2[:,i])
                var[i] = sf2opt - mtimes(mtimes(k.T, invK), k)

            meanfcn = Function('meanfcn', [x], [mean * stdY + meanY])
            # meanfcn2 = Function('meanfcn2',[x,Xnorm2,invKY2],[mean2*stdY + meanY])
            varfcn = Function('varfcn', [x], [var * stdY ** 2])
            # varfcnsd = Function('varfcnsd',[x],[var])

            return meanfcn, varfcn  # , meanfcn2, varfcnsd

    def covSEard(self):
        nx_dim = self.nx_dim
        ell    = SX.sym('ell', nx_dim)
        sf2    = SX.sym('sf2')
        x, z   = SX.sym('x', nx_dim), SX.sym('z', nx_dim)
        dist   = sum1((x - z)**2 / ell)
        covSEfcn = Function('covSEfcn',[x,z,ell,sf2],[sf2*exp(-.5*dist)])

        return covSEfcn


    def GP_inference_np(self, x):
        '''
        --- decription ---
        '''

        nx_dim = self.nx_dim
        kernel, ny_dim = self.kernel, self.ny_dim
        hypopt, Cov_mat = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample = self.calc_cov_sample
        invKsample = self.invKopt
        Xsample, Ysample = self.X_norm, self.Y_norm
        var_out = self.var_out
        if self.GP_casadi:
            if var_out:
                return self.meanfcn(x).toarray().flatten()[0], self.varfcn(x).toarray().flatten()[0]
            else:
                return self.meanfcn(x).toarray().flatten()[0]#.flatten()[0]
        else:
            # Sigma_w                = self.Sigma_w (if input noise)

            xnorm = (x - meanX) / stdX
            mean = np.zeros(ny_dim)
            var = np.zeros(ny_dim)
            # --- Loop over each output (GP) --- #
            for i in range(ny_dim):
                invK = invKsample[i]
                hyper = hypopt[:, i]
                ellopt, sf2opt = np.exp(2 * hyper[:nx_dim]), np.exp(2 * hyper[nx_dim])

                # --- determine covariance of each output --- #
                k = calc_cov_sample(xnorm, Xsample, ellopt, sf2opt)
                mean[i] = np.matmul(np.matmul(k.T, invK), Ysample[:, i])
                var[i] = max(0., sf2opt - np.matmul(np.matmul(k.T, invK), k))  # numerical error
                # var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

            # --- compute un-normalized mean --- #
            mean_sample = mean * stdY + meanY
            var_sample = var * stdY ** 2

            if var_out:
                return mean_sample, var_sample
            else:
                return mean_sample.flatten()[0]

######################################
# Central finite differences 5 points
######################################

def central_finite_diff5(f, x):
    Delta = np.sqrt(np.finfo(float).eps) #step-size is taken as the square root of the machine precision
    n     = np.shape(x)[0]
    x     = x.reshape((n,1))
    dX    = np.zeros((n,1))
    for j in range(n):
        x_d_f2 = np.copy(x); x_d_f1 = np.copy(x)
        x_d_b2 = np.copy(x); x_d_b1 = np.copy(x)

        x_d_f2[j] = x_d_f2[j] + 2*Delta; x_d_b2[j] = x_d_b2[j] - 2*Delta
        x_d_f1[j] = x_d_f1[j] + 1*Delta; x_d_b1[j] = x_d_b1[j] - 1*Delta

        dX[j]     = (f(x_d_b2.flatten()) - 8*f(x_d_b1.flatten()) + 8*f(x_d_f1.flatten()) - f(x_d_f2.flatten()))/(12*Delta)

    return dX

#########################################
# Central second order finite differences
#########################################

def Second_diff_fxx(f, x):
    '''
    Calculating the Hessian via finite differences: https://en.wikipedia.org/wiki/Finite_difference
    '''
    Delta = 1e2*np.sqrt(np.finfo(float).eps) #step-size is taken as function of machine precision
    n     = np.shape(x)[0]
    x     = x.reshape((n,1))
    Hxx   = np.zeros((n,n))
    for j in range(n):
        # compute Fxx (diagonal elements)
        x_d_f    = np.copy(x)
        x_d_b    = np.copy(x)
        x_d_f[j] = x_d_f[j] + Delta
        x_d_b[j] = x_d_b[j] - Delta
        Hxx[j,j] = (f(x_d_f) -2*f(x) + f(x_d_b))/Delta**2

        for i in range(j+1,n):
            # compute Fxy (off-diagonal elements)
            # Fxy
            x_d_fxfy    = np.copy(x_d_f)
            x_d_fxfy[i] = x_d_fxfy[i] + Delta
            x_d_fxby    = np.copy(x_d_f)
            x_d_fxby[i] = x_d_fxby[i] - Delta
            x_d_bxfy    = np.copy(x_d_b)
            x_d_bxfy[i] = x_d_bxfy[i] + Delta
            x_d_bxby    = np.copy(x_d_b)
            x_d_bxby[i] = x_d_bxby[i] - Delta
            Hxx[j,i]    = (f(x_d_fxfy) - f(x_d_fxby) -
                           f(x_d_bxfy) + f(x_d_bxby))/(4*Delta**2)
            Hxx[i,j]    = Hxx[j,i]

    return Hxx


class ITR_GP_RTO:

    ###################################
    # --- initializing ITR_GP_RTO --- #
    ###################################

    def __init__(self, obj_model, obj_system, cons_model, cons_system, x0,
                 Delta0, Delta_max, eta0, eta1, gamma_red, gamma_incr,
                 n_iter, data, bounds, obj_setting=2, noise=None, multi_opt=5, multi_hyper=10, TR_scaling=True,
                 TR_curvature=True, store_data=True, inner_TR=False, scale_inputs=False,GP_casadi=True):
        '''
        data = ['int', bound_list=[[0,10],[-5,5],[3,8]], samples_number] <=> d = ['int', np.array([[-12, 8]]), 3]
        data = ['data0', Xtrain]

        Note 1: remember the data collected

        '''
        # internal variable definitions
        self.GP_casadi = GP_casadi
        self.obj, self.noise = obj_setting, noise
        self.obj_model, self.n_iter = obj_model, n_iter
        self.multi_opt, self.multi_hyper = multi_opt, multi_hyper
        self.store_data, self.data, self.bounds = store_data, data, bounds
        self.x0, self.obj_system, self.cons_model = x0, obj_system, cons_model
        self.cons_system, self.TR_curvature = cons_system, TR_curvature
        self.TR_scaling = TR_scaling
        # TR adjustment variables
        self.Delta_max, self.eta0, self.eta1 = Delta_max, eta0, eta1
        self.gamma_red, self.gamma_incr = gamma_red, gamma_incr
        self.inner_TR = inner_TR
        self.scale_inputs = scale_inputs
        # other definitions
        if scale_inputs:
            self.TRmat = np.linalg.inv(np.diag(bounds[:, 1] - bounds[:, 0]))
        else:
            self.TRmat = np.eye(len(x0))

        self.ng = len(cons_model)
        if noise ==None:
            self.noise = [None]*self.ng
        self.Delta0 = Delta0
        # data creating
        self.Xtrain, self.ytrain = self.data_handling()
        self.ndim, self.ndat = self.Xtrain.shape[1], self.Xtrain.shape[0]
        # alerts
        if TR_curvature == True:
            TR_scaling = True
        print('note: remember constraints are set as positive, so they should be set as -g(x)')

    #########################
    # --- training data --- #
    #########################

    def data_handling(self):
        '''
        --- description ---
        '''
        data = self.data

        # Training data
        if data[0] == 'int':
            print('- No preliminar data supplied, computing data by sobol sequence')
            Xtrain = np.array([])
            Xtrain, ytrain = self.compute_data(data, Xtrain)
            return Xtrain, ytrain

        elif data[0] == 'data0':
            print('- preliminar data supplied, computing objective and constraint values')
            Xtrain = data[1]
            Xtrain, ytrain = self.compute_data(data, Xtrain)
            return Xtrain, ytrain

        else:
            print('- error, data ragument ', data, ' is of wrong type; can be int or ')
            return None

            ##########################

    # --- computing data --- #
    ##########################

    def compute_data(self, data, Xtrain):
        '''
        --- description ---
        '''
        # internal variable calls
        obj_model, cons_model = self.obj_model, self.cons_model
        data[1], cons_system = np.array(data[1]), self.cons_system
        ng, obj_system = self.ng, self.obj_system

        if Xtrain.shape == (0,):  # no data suplied
            # data arrays
            ndim = data[1].shape[0]
            x_max, x_min = data[1][:, 1], data[1][:, 0]
            ndata = data[2]
            Xtrain = np.zeros((ndata, ndim))
            ytrain = np.zeros((ng + 1, ndata))
            funcs_model = [obj_model] + cons_model
            funcs_system = [obj_system] + cons_system

            for ii in range(ng + 1):
                # computing data
                fx = np.zeros(ndata)
                xsmpl = sobol_seq.i4_sobol_generate(ndim, ndata)  # xsmpl.shape = (ndat,ndim)

                # computing Xtrain
                for i in range(ndata):
                    xdat = x_min + xsmpl[i, :] * (x_max - x_min)
                    Xtrain[i, :] = xdat
                for i in range(ndata):
                    fx[i] = funcs_system[ii](np.array(Xtrain[i, :])) - funcs_model[ii](np.array(Xtrain[i, :]))
                # not meant for multi-output
                ytrain[ii, :] = fx

        else:  # data suplied
            # data arrays
            ndim = Xtrain.shape[1]
            ndata = Xtrain.shape[0]
            ytrain = np.zeros((ng + 1, ndata))
            funcs_model = [obj_model] + cons_model
            funcs_system = [obj_system] + cons_system

            for ii in range(ng + 1):
                fx = np.zeros(ndata)

                for i in range(ndata):
                    fx[i] = funcs_system[ii](np.array(Xtrain[i, :])) - funcs_model[ii](np.array(Xtrain[i, :]))
                # not meant for multi-output
                ytrain[ii, :] = fx

        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        return Xtrain.reshape(ndata, ndim, order='F'), ytrain.reshape(ng + 1, ndata, order='F')

    ##############################
    # --- GP as obj function --- #
    ##############################

    def GP_obj_f(self, d, GP, xk):
        '''
        define exploration - explotation strategy
        '''
        obj = self.obj # setting on what the optimizer will do
        obj_model = self.obj_model

        # internal variable calls
        if obj == 1:
            obj_f = GP.GP_inference_np(xk + d)
            return obj_model((xk + d).flatten()) + obj_f[0]

        elif obj == 2:
            obj_f = GP.GP_inference_np(xk + d)
            return obj_model((xk + d).flatten()) + obj_f[0] - 3 * np.sqrt(obj_f[1])
        elif obj == 3:
            fs = self.obj_min#
            obj_f = GP.GP_inference_np(xk + d)
            mean = obj_model((xk + d).flatten()) + obj_f[0]
            Delta = fs - mean
            sigma = np.sqrt(obj_f[1])
            # Delta_p = np.max(mean(X) - fs)
            if sigma == 0.:
                Z = 0.
            else:
                Z = (Delta) / sigma
            EI = -(sigma * norm.pdf(Z) + Delta * norm.cdf(Z))
            return EI  # -(GP.GP_inference_np(X)[0][0] + 3 * GP.GP_inference_np(X)[1][0])#

        else:
            print('error, objective for GP not specified')

    ####################################
    # --- System constraints check --- #
    ####################################

    def System_Cons_check(self, x):
        '''
        This checks if all the constraints of the system are satisfied
        '''
        # internal calls
        cons_system = self.cons_system

        cons = []
        for con_i in range(len(cons_system)):
            cons.append(cons_system[con_i](x))
        cons = np.array(cons)
        satisfact = cons > 0
        satisfact = satisfact.all()

        return satisfact

    ###################################
    # --- Trust Region Constraint --- #
    ###################################

    def TR_con(self, d):
        '''
        TR constraint
        '''
        if self.TR_scaling:
            d = d.flatten()
            d = d / self.Broyd_norm.flatten()

        return self.Delta0 ** 2 - d @ d.T

    #############################################
    # --- Curvature Trust Region Constraint --- #
    #############################################

    def Curvature_TR_con(self, H_norm_vec, d):
        '''
        TR constraint
        '''
        return self.Delta0 ** 2 - d * H_norm_vec @ d.T

    ##############################
    # --- Nearest PSD matrix --- #
    ##############################

    def Nearest_PSDM(self, A_mat):
        '''
        Computes the nearest Symmetric PSD matrix for the Frobenius norm.
        Since Matrix is Symmetric, we could even pass eigenvalues and eigenvecs
        '''
        eps = 1e-4  # notice that there is no minimum, just an infimum. Positive definite matrices are not a closed set.
        B_mat = 0.5 * (A_mat + A_mat.T)  # not necessary if matrix is already symmetric
        eigVals, Q = np.linalg.eig(B_mat)  # eigendecomposition
        eigVals[np.argwhere(eigVals < 0)] = eps  # take eigVals+ = max(eigVals,0)
        L_mat = np.diag(eigVals)
        A_PSD = Q @ L_mat @ Q.T

        return A_PSD

    ###################################
    # --- Mismatch Constraint --- #
    ##################################

    def mistmatch_con(self, xk, GP_con_i, cons_model_i, d):
        '''
        mistmatch constraint
        '''
        return cons_model_i((xk + d).flatten()) + GP_con_i.GP_inference_np((xk + d).flatten())

    ##########################
    # --- Inner TR shape --- #
    ##########################

    def Inner_TR_shape(self, xk, GP_obj, GP_con):
        '''
        THIS IS WRONG, WE SHOULD ADD A THE MODEL INTO THE computation of this
        Shaping the inner TR constraint
        '''
        # inner function calls
        ndim, n_outputs, xk = self.ndim, self.n_outputs, xk.flatten()
        # creting empty matrix
        Q_mat = np.zeros((n_outputs, ndim))
        # obtaining dy/dx vector
        GP_obj.var_out = False
        Q_mat[0, :] = central_finite_diff5(GP_obj.GP_inference_np, xk).flatten()
        Q_mat[0, :] = Q_mat[0, :] / np.sqrt(np.exp(2 * GP_obj.hypopt[ndim + 1]) * GP_obj.Y_std)
        for i_c in range(n_outputs - 1):
            Q_mat[i_c + 1, :] = central_finite_diff5(GP_con[i_c].GP_inference_np, xk).flatten()
            Q_mat[i_c + 1, :] = Q_mat[i_c + 1, :] / np.sqrt(
                np.exp(2 * GP_con[i_c].hypopt[ndim + 1]) * GP_con[i_c].Y_std)

        # V_mat          = np.linalg.inv(Q_mat.T@Q_mat)
        Q_mat = Q_mat.T @ Q_mat
        GP_obj.var_out = True

    ###################################
    # --- Inner Region Constraint --- #
    ###################################

    def Inner_TR_con(self, inner_TR_rad, d):
        '''
        TR inner constraint
        '''
        print('=======  this is not implemented =======')
        return d @ d.T - inner_TR_rad ** 2

    ##########################
    # --- Broyden Update --- #
    ##########################

    def Broyden_update(self, y_new, y_past, x_new, x_past):
        '''
        Updating the gradient by a Broyden update
        '''
        Broyd = self.Broyd
        x_vec = (x_new - x_past).T
        y_vec = (y_new - y_past).T
        Broyd = Broyd + (y_vec - Broyd.T @ x_vec) / (x_vec.T @ x_vec) * x_vec

        return Broyd

    #######################
    # --- TR function --- #
    #######################

    def Adjust_TR(self, Delta0, xk, xnew, GP_obj, i_rto):
        '''
        Adjusts the TR depending on the rho ratio between xk and xnew
        '''
        Delta_max, eta0, eta1 = self.Delta_max, self.eta0, self.eta1
        gamma_red, gamma_incr = self.gamma_red, self.gamma_incr
        obj_system = self.obj_system

        # --- compute rho --- #
        plant_i = obj_system(np.array(xk).flatten())
        plant_iplus = obj_system(np.array(xnew).flatten())
        rho = (plant_i - plant_iplus) / (
                    (self.obj_model((xk).flatten()) + GP_obj.GP_inference_np(np.array(xk).flatten())[0]) -
                    (self.obj_model((xnew).flatten()) + GP_obj.GP_inference_np(np.array(xnew).flatten())[0]))

        # --- Update TR --- #
        if plant_iplus < plant_i:
            if rho >= eta0:
                if rho >= eta1:
                    Delta0 = min(Delta0 * gamma_incr, Delta_max)  # here we should add gradient !!
                elif rho < eta1:
                    Delta0 = Delta0 #* gamma_red
                # Note: xk = xnew this is done later in the code
            if rho < eta0:
                # xnew                    = xk
                # self.backtrack_l[i_rto] = True
                # print('rho<eta0 -- backtracking')
                Delta0 = Delta0 * gamma_red
        else:
            xnew = xk
            Delta0 = Delta0 * gamma_red
            self.backtrack_l[i_rto] = True
            print('plant_iplus<plant_i -- backtracking')
        # if Delta0<0.1:
        #     Delta0 = 0.1
        return Delta0, xnew, xk

    ###################################
    # --- Random Sampling in ball --- #
    ###################################

    def Ball_sampling(self, ndim):
        '''
        This function samples randomly withing a ball of radius self.Delta0
        '''
        u = np.random.normal(0, 1, ndim)  # random sampling in a ball
        norm = np.sum(u ** 2) ** (0.5)
        r = random.random() ** (1.0 / ndim)
        d_init = r * u / norm * self.Delta0 * 2  # random sampling in a ball

        return d_init

    #########################################
    # --- Random Sampling in an ellipse --- #
    #########################################

    def Ellipse_sampling(self, ndim):
        '''
        This function samples randomly withing a ball of radius self.Delta0
        '''
        # u = np.random.normal(0, 1, ndim)  # random sampling in a ball
        # norm = np.sum(u ** 2) ** (0.5)
        # r = random.random() ** (1.0 / ndim)
        # d_init = r * u / norm * self.Delta0**0.5  # random sampling in a ball

        Gamma_Threshold = 0.99  # 9.2103
        S = np.linalg.inv(self.TRmat) * self.Delta0 ** 2
        z_hat = np.zeros(ndim)
        m_FA = 200
        nz = ndim
        z_hat = z_hat.reshape(nz, 1)

        X_Cnz = np.random.normal(size=(nz, m_FA))

        rss_array = np.sqrt(np.sum(np.square(X_Cnz), axis=0))
        kron_prod = np.kron(np.ones((nz, 1)), rss_array)

        X_Cnz = X_Cnz / kron_prod  # Points uniformly distributed on hypersphere surface

        R = np.ones((nz, 1)) * (np.power(np.random.rand(1, m_FA), (1. / nz)))

        unif_sph = R * X_Cnz  # m_FA points within the hypersphere
        T = np.asmatrix(np.linalg.cholesky(S))  # Cholesky factorization of S => S=T’T

        unif_ell = T.H * unif_sph  # Hypersphere to hyperellipsoid mapping

        # Translation and scaling about the center
        z_fa = (unif_ell * np.sqrt(Gamma_Threshold) + (z_hat * np.ones((1, m_FA))))
        for i in range(200):
            if self.TR_con(z_fa[:, i].reshape((1, -1))) >= 0:
                d_init = z_fa[:, i]
                break

        return np.array(d_init).flatten()

    ####################
    # --- inner TR --- #
    ####################

    def Inner_TR_f(self, GP_con, GP_obj):
        '''
        This function samples randomly withing an ellipse self.Delta0
        '''
        ndim = self.ndim
        sn2_l = []

        # compute paraneter for GP_obj
        sn2_l.append(np.exp(2 * GP_obj.hypopt[ndim + 1]) * GP_obj.Y_std)

        for i in range(len(GP_con)):
            sn2_l.append(np.exp(2 * GP_con[i].hypopt[ndim + 1]) * GP_con[i].Y_std)

        print('max(sn2_l) = ', max(sn2_l)[0])

        return max(sn2_l)[0]

    ########################################
    # --- Constrain Violation and step --- #
    ########################################

    def Step_constraint(self, Delta0, xk, xnew, GP_obj, i_rto):
        '''
        Calls Adjust_TR which adjusts the trust region and decides on the step size
        depending on constraint violation or the objective similarity
        '''
        Adjust_TR = self.Adjust_TR

        if not self.System_Cons_check(np.array(xnew).flatten()):
            xnew = xk
            self.backtrack_l[i_rto] = True
            print('Constraint violated -- backtracking')
            Delta0 = Delta0 * self.gamma_red

            # if Delta0 < 0.1:
            #     Delta0 = 0.1

            return Delta0 , xnew, xk

        else:
            Delta0, xnew, xk = Adjust_TR(Delta0, xk, xnew, GP_obj, i_rto)
            return Delta0, xnew, xk

    ########################################
    # --- Constraint GP construction --- #
    ########################################

    def GPs_construction(self, xk, Xtrain, ytrain, ndatplus, H_norm=0.):
        '''
        Constructs a GP for every cosntraint
        '''
        # internal calls
        ndat, multi_hyper, ng, ndim = self.ndat + ndatplus, self.multi_hyper, self.ng, self.ndim
        cons_model, mistmatch_con = self.cons_model, self.mistmatch_con
        TR_curvature, Curvature_TR_con = self.TR_curvature, self.Curvature_TR_con

        # --- objective function GP --- #
        self.obj_max = np.max(ytrain[0, :].reshape(ndat, 1))
        if self.noise[0] == None:
            GP_obj = GP_model(Xtrain, ytrain[0, :].reshape(ndat, 1), 'RBF',
                          multi_hyper=multi_hyper, var_out=True)
        else:
            GP_obj = GP_model(Xtrain, ytrain[0, :].reshape(ndat, 1), 'RBF',
                          multi_hyper=multi_hyper, var_out=True, noise = self.noise[0])
        GP_con = [0] * ng  # Gaussian processes that output mistmatch (functools)
        GP_con_2 = [0] * ng  # Constraints for the NLP

        if TR_curvature:
            GP_con_curv = [0] * (ndim)  # TR ellipsoid (functools)
            GP_con_f = [0] * (ng + ndim)  # Constraint plus ellipsoids
        else:
            GP_con_f = [0] * (ng + 1)

        for igp in range(ng):
            if self.noise[0] == None:
                GP_con[igp] = GP_model(Xtrain, ytrain[igp + 1, :].reshape(ndat, 1), 'RBF',
                                   multi_hyper=multi_hyper, var_out=False)
            else:
                GP_con[igp] = GP_model(Xtrain, ytrain[igp + 1, :].reshape(ndat, 1), 'RBF',
                                   multi_hyper=multi_hyper, var_out=False, noise = self.noise[igp+1])

            GP_con_2[igp] = functools.partial(mistmatch_con, xk, GP_con[igp],
                                              cons_model[igp])  # partially evaluating a function
            GP_con_f[igp] = {'type': 'ineq', 'fun': GP_con_2[igp]}
        if TR_curvature:
            for jdim in range(ndim):
                GP_con_curv[jdim] = functools.partial(Curvature_TR_con, H_norm[:, jdim].reshape(1, ndim))
                GP_con_f[ng + jdim] = {'type': 'ineq', 'fun': GP_con_curv[jdim]}
        else:
            GP_con_f[igp + 1] = {'type': 'ineq', 'fun': self.TR_con}
        if self.inner_TR:
            inner_TR_val = Inner_TR_f(GP_con, GP_obj)
            inner_TR_ff = functools.partial(self.Inner_TR_con, self.inner_TR_val)
            GP_con_f.append({'type': 'ineq', 'fun': inner_TR_ff})

        return GP_obj, GP_con_f, GP_con

    #############################################################
    # --- Update derivatives and Hessian (Broyden and BFGS) --- #
    #############################################################

    def Update_derivatives(self, xk, xnew, H_inv, H_norm):
        '''
        H_norm is passed in case xk==xnew, then the old value of H_norm is passed back
        '''
        # internal function calls
        TR_curvature, obj_system = self.TR_curvature, self.obj_system
        Broyden_update, Nearest_PSDM = self.Broyden_update, self.Nearest_PSDM

        # --- Broyden update --- #
        if np.sum(np.abs(xk)) != np.sum(np.abs(xnew)):
            y_iplus = obj_system(np.array(xnew[:]).flatten())
            y_i = obj_system(np.array(xk[:]).flatten())
            Broyd_past = np.copy(self.Broyd)
            self.Broyd = Broyden_update(y_iplus, y_i, xnew, xk)
            # --- BFGS for the inverse Hessian approx --- #
            skk = (xnew - xk).T
            ykk = self.Broyd - Broyd_past
            rhokk = 1. / (ykk.T @ skk)
            Imatrix = np.eye(skk.shape[0])
            H_inv = (Imatrix - rhokk * skk @ ykk.T) @ H_inv @ (Imatrix - rhokk * ykk @ skk.T) + rhokk * skk @ skk.T
            if not np.all(np.linalg.eigvals(H_inv) >= 0):
                print('H_inv not PSD, making it PSD')
                H_inv = Nearest_PSDM(H_inv)
            # --- BFGS for the Hessian approx --- #
            self.Broyd_norm = H_inv @ self.Broyd
            Broyd_2norm = np.linalg.norm(self.Broyd_norm)
            self.Broyd_norm = self.Broyd_norm / Broyd_2norm
            if TR_curvature == True:  # if ellipse TR WITH rotation
                H_matrix = np.linalg.inv(H_inv)
                H_det = np.linalg.det(H_matrix)
                H_norm = H_matrix / H_det
                # print('PSD H_norm? ',np.all(np.linalg.eigvals(H_norm) >= 0))
                # print('PSD H_inv? ',np.all(np.linalg.eigvals(H_inv) >= 0))
        else:
            print('xk == xnew')

        return H_norm, H_inv


    def find_min_so_far(self,funcs_system, X_opt):
            # ynew[0,ii] = funcs[ii](np.array(xnew[:]).flatten())
        min = np.inf
        for i in range(len(X_opt)):
                y= funcs_system[0](np.array(X_opt[i]).flatten())
                if y< min:
                    min =y
        return min
    ######################################
    # --- Real-Time Optimization alg --- #
    ######################################

    def RTO_routine(self):
        '''
        --- description ---
        '''
        # internal variable calls
        obj_model, cons_model = self.obj_model, self.cons_model
        store_data, Xtrain, ytrain = self.store_data, self.Xtrain, self.ytrain
        multi_hyper, n_iter = self.multi_hyper, self.n_iter
        ndim, ndat, multi_opt = self.ndim, self.ndat, self.multi_opt
        GP_obj_f, bounds, ng = self.GP_obj_f, self.bounds, self.ng
        multi_opt, Delta0, x0 = self.multi_opt, self.Delta0, self.x0
        obj_system, cons_system = self.obj_system, self.cons_system
        Adjust_TR, Step_constraint = self.Adjust_TR, self.Step_constraint
        TR_curvature, mistmatch_con = self.TR_curvature, self.mistmatch_con
        TR_scaling, Broyden_update = self.TR_scaling, self.Broyden_update
        Curvature_TR_con, Ball_sampling = self.Curvature_TR_con, self.Ball_sampling
        inner_TR, Inner_TR_f, = self.inner_TR, self.Inner_TR_f
        Inner_TR_con, GPs_construction = self.Inner_TR_con, self.GPs_construction
        Update_derivatives = self.Update_derivatives
        self.n_outputs, Inner_TR_shape = len(cons_model) + 1, self.Inner_TR_shape

        # variable definitions
        funcs_model = [obj_model] + cons_model
        funcs_system = [obj_system] + cons_system
        self.backtrack_l = [False] * n_iter  # !!

        # Initialize B~dF/dx and H~dF^2/dx^2 as gradient and derivatives from the model
        if TR_scaling:
            self.Broyd = central_finite_diff5(obj_model, x0)
            H_matrix = Second_diff_fxx(obj_model, x0)
            H_inv = np.linalg.inv(H_matrix)
            step_dir = H_inv @ self.Broyd
            step_dir_norm = np.linalg.norm(step_dir)
            self.Broyd_norm = step_dir / step_dir_norm
            if TR_curvature:  # if ellipse TR WITH rotation
                H_det = np.linalg.det(H_matrix)
                H_norm = H_matrix / H_det

        # --- building GP models from existing data --- #
        # evaluating initial point
        xnew = x0
        Xtrain = np.vstack((Xtrain, xnew))
        ynew = np.zeros((1, ng + 1))
        for ii in range(ng + 1):
            ynew[0, ii] = funcs_system[ii](np.array(xnew[:]).flatten()) - funcs_model[ii](np.array(xnew[:]).flatten())
        ytrain = np.hstack((ytrain, ynew.T))
        # --- building GP models for first time --- #
        if TR_scaling and TR_curvature:
            GP_obj, GP_con_f, GP_con = GPs_construction(xnew, Xtrain, ytrain, 1, H_norm)
        else:
            GP_obj, GP_con_f, GP_con = GPs_construction(xnew, Xtrain, ytrain, 1)

        # renaming data
        X_opt = np.copy(Xtrain)
        y_opt = np.copy(ytrain)
        self.obj_min = self.find_min_so_far(funcs_system, X_opt)

        # --- TR lists --- #
        TR_l = ['error'] * (n_iter + 1)
        if TR_curvature:
            TR_l_angle = ['error'] * (n_iter + 1)
            w, v = np.linalg.eig(H_inv)
            max_idx = np.argmax(w)
            TR_l_angle[0] = np.arctan(v[max_idx][1] / v[max_idx][0]) * 180. / np.pi
            TR_l[0] = (2. * self.Delta0 / np.sqrt(np.abs(np.diag(H_norm)))).tolist()
        elif TR_scaling:
            TR_l[0] = (2. * self.Delta0 * self.Broyd_norm).tolist()
        else:
            TR_l[0] = self.Delta0

        # --- rto -- iterations --- #
        options = {'disp': False, 'maxiter': 10000}  # solver options
        lb, ub = bounds[:, 0], bounds[:, 1]
        for i_rto in range(n_iter):
            print('============================ RTO iteration ', i_rto)

            # --- optimization -- multistart --- #
            localsol = [0.] * multi_opt  # values for multistart
            localval = np.zeros((multi_opt))  # variables for multistart
            xk = xnew
            TRb = (bounds - xk.reshape(ndim, 1, order='F'))
            # TODO: sampling on ellipse, not only Ball
            for j in range(multi_opt):
                if TR_curvature or self.scale_inputs:
                    # d_init = Ellipse_sampling(ndim)
                    d_init = self.Ellipse_sampling(ndim)
                else:
                    d_init = self.Ball_sampling(ndim)  # random sampling in a ball

                # GP optimization
                res = minimize(GP_obj_f, d_init, args=(GP_obj, xk), method='SLSQP',
                               options=options, bounds=(TRb), constraints=GP_con_f, tol=1e-12)
                localsol[j] = res.x
                if res.success == True:
                    localval[j] = res.fun
                else:
                    localval[j] = np.inf
            if np.min(localval) == np.inf:
                print('warming, no feasible solution found')

                xnew = xk*(1+0.01*np.random.randn())  # selecting best solution
            else:
            # selecting best point
                minindex = np.argmin(localval)  # choosing best solution
                xnew = localsol[minindex] + xk  # selecting best solution

            # re-evaluate best point (could be done more efficiently - no re-evaluation)
            xnew = np.array([xnew]).reshape(1, ndim)
            ynew = np.zeros((1, ng + 1))
            for ii in range(ng + 1):
                # ynew[0,ii] = funcs[ii](np.array(xnew[:]).flatten())
                ynew[0, ii] = funcs_system[ii](np.array(xnew[:]).flatten()) - funcs_model[ii](
                    np.array(xnew[:]).flatten())
            print('New Objective: ', -funcs_system[0](np.array(xnew[:]).flatten()))
            # adding new point to sample
            X_opt = np.vstack((X_opt, xnew))
            y_opt = np.hstack((y_opt, ynew.T))
            self.obj_min = self.find_min_so_far(funcs_system, X_opt)

            # --- Update TR --- #
            self.Delta0, xnew, xk = Step_constraint(self.Delta0, xk, xnew, GP_obj, i_rto)  # adjust TR

            # --- Update derivatives and Hessian (Broyden and BFGS) --- #
            if TR_scaling:
                H_norm, H_inv = Update_derivatives(xk, xnew, H_inv, H_norm)

            # --- re-training GP --- #
            if TR_scaling and TR_curvature:
                GP_obj, GP_con_f, GP_con = GPs_construction(xnew, X_opt, y_opt, 2 + i_rto, H_norm)
            else:
                GP_obj, GP_con_f, GP_con = GPs_construction(xnew, X_opt, y_opt, 2 + i_rto)

            # --- TR listing --- #
            if not TR_scaling:  # if normal TR
                TR_l[i_rto + 1] = self.Delta0
                if inner_TR:
                    print('ratio of inner to outer TR = ', self.Delta0 / (inner_TR_val + 1e-8))
            else:
                if TR_curvature == True:  # if ellipse TR WITH rotation
                    TR_l[i_rto + 1] = (2. * self.Delta0 / np.sqrt(np.abs(np.diag(H_norm)))).tolist()
                    w, v = np.linalg.eig(H_matrix)
                    max_idx = np.argmax(w)
                    TR_l_angle[i_rto + 1] = np.arctan(v[max_idx][1] / v[max_idx][0]) * 180. / np.pi
                else:
                    TR_l[i_rto + 1] = (2. * self.Delta0 * self.Broyd_norm).tolist()

            # Inner_TR_shape(xk, GP_obj, GP_con)
        # --- output data --- #
        if not TR_curvature:
            return X_opt, y_opt, TR_l, xnew, self.backtrack_l
        else:
            return X_opt, y_opt, TR_l, TR_l_angle, xnew, self.backtrack_l