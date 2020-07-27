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

from casadi import *


def Benoit_Model(u):
    f = u[0] ** 2 + u[1] ** 2
    return f


def con1_model(u):
    g1 = 1. - u[0] + u[1] ** 2
    return -g1


def Benoit_System(u):
    f = u[0] ** 2 + u[1] ** 2 + u[0] * u[1] + np.random.normal(0., np.sqrt(1e-3))
    return f


def con1_system(u):
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] - 2. + np.random.normal(0., np.sqrt(1e-3))
    return -g1


def con1_system_tight(u):
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] + np.random.normal(0., np.sqrt(1e-3))
    return -g1


def Benoit_System_noiseless(u):
    f = u[0] ** 2 + u[1] ** 2 + u[0] * u[1]  # + np.random.normal(0., np.sqrt(1e-3))
    return f


def con1_system_noiseless(u):
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1] - 2.  # + np.random.normal(0., np.sqrt(1e-3))
    return -g1


def con1_system_tight_noiseless(u):
    g1 = 1. - u[0] + u[1] ** 2 + 2. * u[1]  # + np.random.normal(0., np.sqrt(1e-3))
    return -g1


class WO_system:
    # Parameters
    Fa = 1.8275
    Mt = 2105.2
    # kinetic parameters
    phi1 = - 3.
    psi1 = -17.
    phi2 = - 4.
    psi2 = -29.
    # Reference temperature
    Tref = 110. + 273.15  # [=] K.

    def __init__(self):
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.states, self.algebraics, self.inputs = self.DAE_system()
        self.eval = self.integrator_system()

    def DAE_system(self):
        # Define vectors with names of states
        states = ['x']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = ['Xa', 'Xb', 'Xc', 'Xe', 'Xp', 'Xg']
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        inputs = ['Fb', 'Tr']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Reparametrization
        k1 = 1.6599e6 * np.exp(-6666.7 / (Tr + 273.15))
        k2 = 7.2117e8 * np.exp(-8333.3 / (Tr + 273.15))
        k3 = 2.6745e12 * np.exp(-11111. / (Tr + 273.15))

        # reaction rate
        Fr = Fa + Fb
        r1 = k1 * Xa * Xb * Mt
        r2 = k2 * Xb * Xc * Mt
        r3 = k3 * Xc * Xp * Mt

        # residual for x
        x_res = np.zeros((6, 1))
        x_res[0, 0] = (Fa - r1 - Fr * Xa) / Mt
        x_res[1, 0] = (Fb - r1 - r2 - Fr * Xb) / Mt
        x_res[2, 0] = (+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt
        x_res[3, 0] = (+ 2 * r2 - Fr * Xe) / Mt
        x_res[4, 0] = (+   r2 - 0.5 * r3 - Fr * Xp) / Mt
        x_res[5, 0] = (+ 1.5 * r3 - Fr * Xg) / Mt
        # Define vectors with banes of input variables

        ODEeq = [0 * x]

        # Declare algebraic equations
        Aeq = []

        Aeq += [(Fa - r1 - Fr * Xa) / Mt]
        Aeq += [(Fb - r1 - r2 - Fr * Xb) / Mt]
        Aeq += [(+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt]
        Aeq += [(+ 2 * r2 - Fr * Xe) / Mt]
        Aeq += [(+   r2 - 0.5 * r3 - Fr * Xp) / Mt]
        Aeq += [(+ 1.5 * r3 - Fr * Xg) / Mt]

        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs

    def integrator_system(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = self.DAE_system()
        VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
        solver = rootfinder('solver', 'newton', VV)

        return solver

    def WO_obj_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        Fb = u[0]
        Tr = u[1]
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(1043.38 * x[4] * Fr +
                20.92 * x[3] * Fr -
                79.23 * Fa -
                118.34 * Fb) + 0.5 * np.random.normal(0., 1)

        return obj

    def WO_obj_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        Fb = u[0]
        Tr = u[1]
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(1043.38 * x[4] * Fr +
                20.92 * x[3] * Fr -
                79.23 * Fa -
                118.34 * Fb)  # + 0.5*np.random.normal(0., 1)

        return obj

    def WO_con1_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon1 = x[0] - 0.12 + 5e-4 * np.random.normal(0., 1)

        return -pcon1.toarray()[0]

    def WO_con2_sys_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon2 = x[5] - 0.08 + 5e-4 * np.random.normal(0., 1)

        return -pcon2.toarray()[0]

    def WO_con1_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(0., 1)

        return -pcon1.toarray()[0]

    def WO_con2_sys_ca_noise_less(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.0260265, 0.207296, 0.0923376, 0.0339309]), u)
        pcon2 = x[5] - 0.08  # + 5e-4*np.random.normal(0., 1)

        return -pcon2.toarray()[0]


class WO_model:
    # Parameters
    Fa = 1.8275
    Mt = 2105.2
    # kinetic parameters
    phi1 = - 3.
    psi1 = -17.
    phi2 = - 4.
    psi2 = -29.
    # Reference temperature
    Tref = 110. + 273.15  # [=] K.

    def __init__(self):
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.states, self.algebraics, self.inputs = self.DAE_model()
        self.eval = self.integrator_model()

    def DAE_model(self):
        # Define vectors with names of states
        states = ['x']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = ['Xa', 'Xb', 'Xe', 'Xp', 'Xg']
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        # Define vectors with banes of input variables
        inputs = ['Fb', 'Tr']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        k1 = np.exp(phi1) * np.exp((Tref / (Tr + 273.15) - 1) * psi1)
        k2 = np.exp(phi2) * np.exp((Tref / (Tr + 273.15) - 1) * psi2)

        # reaction rate
        Fr = Fa + Fb
        r1 = k1 * Xa * Xb * Xb * Mt
        r2 = k2 * Xa * Xb * Xp * Mt
        ODEeq = [0 * x]

        # Declare algebraic equations
        Aeq = []

        Aeq += [Fa - r1 - r2 - Fr * Xa]
        Aeq += [Fb - 2 * r1 - r2 - Fr * Xb]
        Aeq += [+ 2 * r1 - Fr * Xe]
        Aeq += [+   r1 - r2 - Fr * Xp]
        Aeq += [+ 3 * r2 - Fr * Xg]

        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs

    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = self.DAE_model()
        VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
        solver = rootfinder('solver', 'newton', VV)

        # model = functools.partial(solver, np.zeros(np.shape(xa)))
        return solver

    def WO_obj_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
        Fb = u[0]
        Tr = u[1]
        Fa = 1.8275
        Fr = Fa + Fb

        obj = -(1043.38 * x[3] * Fr +
                20.92 * x[2] * Fr -
                79.23 * Fa -
                118.34 * Fb)

        return obj

    def WO_con1_model_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
        pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(1., 1)
        return -pcon1.toarray()[0]

    def WO_con2_model_ca(self, u):
        x = self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
        pcon2 = x[4] - 0.08  # + 5e-4*np.random.normal(1., 1)
        return -pcon2.toarray()[0]


def con_empty(u):
    g1 = 0.
    return -g1


def obj_empty(u):
    f = 0.
    return f


class Bio_system:


    def __init__(self):
        self.nk, self.tf, self.x0, _, _ = self.specifications()
        self.xd, self.xa, self.u, _, self.ODEeq, self.Aeq, self.u_min, self.u_max,\
        self.states, self.algebraics, self.inputs, self.nd, self.na, self.nu, \
        self.nmp,self. modparval= self.DAE_system()
        self.eval = self.integrator_model()
        self.Sigma_v = [400.,1e5,1e-2]*diag(np.ones(self.nd))*1e-7
    def specifications(self):
        ''' Specify Problem parameters '''
        tf = 240.  # final time
        nk = 12  # sampling points
        x0 = np.array([1., 150., 0.])
        Lsolver = 'mumps'  # 'ma97'  # Linear solver
        c_code = False  # c_code

        return nk, tf, x0, Lsolver, c_code

    def DAE_system(self):
        # Define vectors with names of states
        states = ['x', 'n', 'q']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = []
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        # Define vectors with banes of input variables
        inputs = ['L', 'Fn']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Define model parameter names and values
        modpar = ['u_m', 'k_s', 'k_i', 'K_N', 'u_d', 'Y_nx', 'k_m', 'k_sq',
                  'k_iq', 'k_d', 'K_Np']
        modparval = [0.0923 * 0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
                     2.544 * 0.62 * 1e-4, 23.51, 800.0, 0.281, 16.89]

        nmp = len(modpar)
        for i in range(nmp):
            globals()[modpar[i]] = SX(modparval[i])

        # Additive measurement noise
        #    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Additive disturbance noise
        #    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Initial additive disturbance noise
        #    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3

        # Declare ODE equations (use notation as defined above)

        dx = u_m * L / (L + k_s + L ** 2. / k_i) * x * n / (n + K_N) - u_d * x
        dn = - Y_nx * u_m * L / (L + k_s + L ** 2. / k_i) * x * n / (n + K_N) + Fn
        dq = k_m * L / (L + k_sq + L ** 2. / k_iq) * x - k_d * q / (n + K_Np)

        ODEeq = [dx, dn, dq]

        # Declare algebraic equations
        Aeq = []

        # Define control bounds
        u_min = np.array([120., 0.])  # lower bound of inputs
        u_max = np.array([400., 40.])  # upper bound of inputs

        # Define objective to be minimized
        t = SX.sym('t')

        return xd, xa, u, 0, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval

    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: NaN
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval \
            = self.DAE_system()

        dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
        opts = {'tf': self.tf / self.nk}  # interval length
        F = integrator('F', 'idas', dae, opts)
        # model = functools.partial(solver, np.zeros(np.shape(xa)))
        return F

    def bio_obj_ca(self, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2))
        u  = u1 * (self.u_max - self.u_min) + self.u_min



        for i in range(self.nk):
            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))#self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
            x = np.array(xd['xf'].T)[0]


        return -x[-1] + np.random.multivariate_normal([0.]*self.nd,np.array(self.Sigma_v))[-1]

    def bio_con1_ca(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2))
        u  = u1 * (self.u_max - self.u_min) + self.u_min



        for i in range(n):
            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))#self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
            x = np.array(xd['xf'].T)[0]
        x[1] += np.random.multivariate_normal([0.]*self.nd,np.array(self.Sigma_v))[1]
        pcon1 = x[1]/800  - 1
        return -pcon1#.toarray()[0]

    def bio_con2_ca(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 *(self.u_max - self.u_min) + self.u_min



        for i in range(n):
            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))#self.eval(np.array([0.114805, 0.525604, 0.207296, 0.0923376, 0.0339309]), u)
            x = np.array(xd['xf'].T)[0]
        x += np.random.multivariate_normal([0.]*self.nd,np.array(self.Sigma_v))
        pcon1 = x[2]/(0.011 * x[0])-1
        return -pcon1#.toarray()[0]



class Bio_model:


    def __init__(self, empty =False):
        self.nk, self.tf, self.x0, _, _ = self.specifications()
        self.xd, self.xa, self.u, _, self.ODEeq, self.Aeq, self.u_min, self.u_max,\
        self.states, self.algebraics, self.inputs, self.nd, self.na, self.nu, \
        self.nmp,self. modparval= self.DAE_system()
        self.eval = self.integrator_model()
        self.empty = empty
    def specifications(self):
        ''' Specify Problem parameters '''
        tf = 240.  # final time
        nk = 12  # sampling points
        x0 = np.array([1., 150., 0.])
        Lsolver = 'mumps'  # 'ma97'  # Linear solver
        c_code = False  # c_code

        return nk, tf, x0, Lsolver, c_code

    def DAE_system(self):
        # Define vectors with names of states
        states = ['x', 'n', 'q']

        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = []
        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        # Define vectors with banes of input variables
        inputs = ['L', 'Fn']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Define model parameter names and values
        modpar = ['u_m', 'k_s', 'k_i', 'K_N', 'u_d', 'Y_nx', 'k_m', 'k_sq',
                  'k_iq', 'k_d', 'K_Np']
        modparval = [0.0923 * 0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
                     2.544 * 0.62 * 1e-4, 23.51, 800.0, 0.281, 16.89]

        nmp = len(modpar)
        for i in range(nmp):
            globals()[modpar[i]] = SX(modparval[i])

        # Additive measurement noise
        #    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Additive disturbance noise
        #    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Initial additive disturbance noise
        #    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3

        # Declare ODE equations (use notation as defined above)

        dx = u_m * L / (L + k_s) * x * n / (n + K_N) - u_d * x
        dn = - Y_nx * u_m * L / (L + k_s) * x * n / (n + K_N) + Fn
        dq = k_m * L / (L + k_sq) * x - k_d * q / (n + K_Np)

        ODEeq = [dx, dn, dq]

        # Declare algebraic equations
        Aeq = []

        # Define control bounds
        u_min = np.array([120., 0.])  # lower bound of inputs
        u_max = np.array([400., 40.])  # upper bound of inputs

        # Define objective to be minimized
        t = SX.sym('t')

        return xd, xa, u, 0, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval

    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: NaN
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval \
            = self.DAE_system()
        ODEeq_ = vertcat(*ODEeq)

        self.ODEeq = Function('f', [xd, u], [vertcat(*ODEeq)], ['x0', 'p'], ['xdot'])

        dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
        opts = {'tf': self.tf / self.nk}  # interval length
        F = integrator('F', 'idas', dae, opts)
        # model = functools.partial(solver, np.zeros(np.shape(xa)))
        return F

    def bio_obj_ca(self, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min



        for i in range(self.nk):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.

            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))
            x = np.array(xd['xf'].T)[0]
            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0


        return -x[-1]

    def bio_con1_ca(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2))
        u  = u1 * (self.u_max - self.u_min) + self.u_min



        for i in range(n):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.


            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))
            x = np.array(xd['xf'].T)[0]
            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[1]/800-1  # + 5e-4*np.random.normal(1., 1)
        return -pcon1#.toarray()[0]

    def bio_con2_ca(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min



        for i in range(n):

            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.

            xd = self.eval(x0=vertcat(np.array(x)), p=vertcat(u[i]))
            x = np.array(xd['xf'].T)[0]
            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[2]/(0.011 * x[0])-1  # + 5e-4*np.random.normal(1., 1)
        return -pcon1#.toarray()[0]

    def bio_obj_ca_RK4(self, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min
        DT = self.tf/self.nk/4


        for i in range(self.nk):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.

            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


            # xd = self.eval(x0=vertcat(np.array(x1)), p=vertcat(u[i]))
            # x1 = np.array(xd['xf'].T)[0]
            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0


        return -x[-1].toarray()[0][0]

    def bio_con1_ca_RK4(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min


        DT = self.tf/self.nk/4

        for i in range(n):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.


            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[1]/800 -1 # + 5e-4*np.random.normal(1., 1)
        return -pcon1.toarray()[0][0]

    def bio_con2_ca_RK4(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min


        DT = self.tf/self.nk/4

        for i in range(n):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.


            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[2]/(0.011 * x[0])-1  # + 5e-4*np.random.normal(1., 1)
        return -pcon1.toarray()[0][0]

    def bio_model_ca(self):
        M = 4  # RK4 steps per interval

        X0 = SX.sym('X0', self.nd)
        U = SX.sym('U', self.nu,1)
        u  = U * (self.u_max - self.u_min) + self.u_min
        DT = self.tf/self.nk/M

        f = self.ODEeq
        X = X0
        for j in range(M):
            k1 = f(X, u)
            k2 = f(X + DT / 2 * k1, u)
            k3 = f(X + DT / 2 * k2, u)
            k4 = f(X + DT * k2, u)
            X  = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        F = Function('F', [X0, U], [X], ['x0', 'u'], ['xf'])

        return F


    def bio_obj_ca_f(self, x):
        if not(self.empty):
            return -x[-1]
        else:
            return 0.

    def bio_con1_ca_f(self, x):
        if not(self.empty):
            pcon1 = x[1]/800 -1 # + 5e-4*np.random.normal(1., 1)
            return -pcon1
        else:
            return 0.

    def bio_con2_ca_f(self, x):
        if not(self.empty):
            pcon1 = x[2]/(0.011 * x[0])-1  # + 5e-4*np.random.normal(1., 1)
            return -pcon1
        else:
            return 0.




    def bio_obj_ca_RK4_empty(self, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min
        DT = self.tf/self.nk/4


        for i in range(self.nk):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.

            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


            # xd = self.eval(x0=vertcat(np.array(x1)), p=vertcat(u[i]))
            # x1 = np.array(xd['xf'].T)[0]
            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0


        return -0*x[-1].toarray()[0][0]

    def bio_con1_ca_RK4_empty(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min


        DT = self.tf/self.nk/4

        for i in range(n):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.


            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[1]/800 -1 # + 5e-4*np.random.normal(1., 1)
        return -0*pcon1.toarray()[0][0]

    def bio_con2_ca_RK4_empty(self, n, u0):
        x  = self.x0
        u1 = np.array(u0).reshape((self.nk,2) )
        u  = u1 * (self.u_max - self.u_min) + self.u_min


        DT = self.tf/self.nk/4

        for i in range(n):
            if np.any(x<0):
                print(2)
            elif np.any(u[i]<0):
                print(2)
            for j in range(self.nk):
                if u[j,1]<0:
                    u[j,1]= 0.


            f = self.ODEeq

            for j in range(4):
                k1 = f(x0=vertcat(np.array(x)), p=vertcat(u[i]))['xdot']
                k2 = f(x0=vertcat(np.array(x + DT / 2 * k1)),p=vertcat(u[i]))['xdot']
                k3 = f(x0=vertcat(np.array(x + DT / 2 * k2)), p=vertcat(u[i]))['xdot']
                k4 = f(x0=vertcat(np.array(x + DT * k2)), p= vertcat(u[i]))['xdot']
                x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            for j in range(self.nd):
                if x[j]<0:
                    x[j]=0
        pcon1 = x[2]/(0.011 * x[0])-1  # + 5e-4*np.random.normal(1., 1)
        return -0*pcon1.toarray()[0][0]

    def bio_model_ca_empty(self):
        M = 4  # RK4 steps per interval

        X0 = SX.sym('X0', self.nd)
        U = SX.sym('U', self.nu,1)
        u  = U * (self.u_max - self.u_min) + self.u_min
        DT = self.tf/self.nk/M

        f = self.ODEeq
        X = X0
        for j in range(M):
            k1 = f(X, u)
            k2 = f(X + DT / 2 * k1, u)
            k3 = f(X + DT / 2 * k2, u)
            k4 = f(X + DT * k2, u)
            X  = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        F = Function('F', [X0, U], [X], ['x0', 'u'], ['xf'])

        return F


    def bio_obj_ca_f_empty(self, x):

        return -0*x[-1]

    def bio_con1_ca_f_empty(self, x):
        pcon1 = x[1]/800 -1 # + 5e-4*np.random.normal(1., 1)
        return -0*pcon1

    def bio_con2_ca_f_empty(self, x):
        pcon1 = x[2]/(0.011 * x[0])-1  # + 5e-4*np.random.normal(1., 1)
        return -0*pcon1