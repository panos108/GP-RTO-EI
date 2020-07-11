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


#
# def DAE_model():
#     # Parameters
#     Fa = 1.8275
#     Mt = 2105.2
#     # kinetic parameters
#     phi1 = - 3.
#     psi1 = -17.
#     phi2 = - 4.
#     psi2 = -29.
#     # Reference temperature
#     Tref = 110. + 273.15  # [=] K.
#     # Define vectors with names of states
#     states = ['x']
#     nd = len(states)
#     xd = SX.sym('xd', nd)
#     for i in range(nd):
#         globals()[states[i]] = xd[i]
#
#     # Define vectors with names of algebraic variables
#     algebraics = ['Xa', 'Xb', 'Xe', 'Xp', 'Xg']
#     na = len(algebraics)
#     xa = SX.sym('xa', na)
#     for i in range(na):
#         globals()[algebraics[i]] = xa[i]
#
#     # Define vectors with banes of input variables
#     inputs = ['Fb', 'Tr']
#     nu = len(inputs)
#     u = SX.sym("u", nu)
#     for i in range(nu):
#         globals()[inputs[i]] = u[i]
#
#     k1 = np.exp(phi1) * np.exp((Tref / (Tr + 273.15) - 1) * psi1)
#     k2 = np.exp(phi2) * np.exp((Tref / (Tr + 273.15) - 1) * psi2)
#
#     # reaction rate
#     Fr = Fa + Fb
#     r1 = k1 * Xa * Xb * Xb * Mt
#     r2 = k2 * Xa * Xb * Xp * Mt
#     ODEeq = [0 * x]
#
#     # Declare algebraic equations
#     Aeq = []
#
#     Aeq += [Fa - r1 - r2 - Fr * Xa]
#     Aeq += [Fb - 2 * r1 - r2 - Fr * Xb]
#     Aeq += [+ 2 * r1 - Fr * Xe]
#     Aeq += [+   r1 - r2 - Fr * Xp]
#     Aeq += [+ 3 * r2 - Fr * Xg]
#
#     return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs
#
#
# def integrator_model():
#     """
#     This function constructs the integrator to be suitable with casadi environment, for the equations of the model
#     and the objective function with variable time step.
#     inputs: NaN
#     outputs: F: Function([x, u, dt]--> [xf, obj])
#     """
#
#     xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = DAE_model()
#     VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
#     solver = rootfinder('solver', 'newton', VV)
#
#     # model = functools.partial(solver, np.zeros(np.shape(xa)))
#     return solver
#
#
# def WO_obj_ca(u):
#     solver = integrator_model()
#     x = solver(np.zeros(5), u)
#     Fb = u[0]
#     Tr = u[1]
#     Fa = 1.8275
#     Fr = Fa + Fb
#
#     obj = -(1043.38 * x[3] * Fr +
#             20.92 * x[2] * Fr -
#             79.23 * Fa -
#             118.34 * Fb)
#
#     return obj
#
#
# def WO_con1_model_ca(u):
#     solver = integrator_model()
#     x = solver(np.zeros(5), u)
#     pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(1., 1)
#     return -pcon1.toarray()[0]
#
#
# def WO_con2_model_ca(u):
#     solver = integrator_model()
#     x = solver(np.zeros(5), u)
#     pcon2 = x[4] - 0.08  # + 5e-4*np.random.normal(1., 1)
#     return -pcon2.toarray()[0]
#
#     # Parameters
#
#
#
#
# def DAE_system():
#     Fa = 1.8275
#     Mt = 2105.2
#     # kinetic parameters
#     phi1 = - 3.
#     psi1 = -17.
#     phi2 = - 4.
#     psi2 = -29.
#     # Reference temperature
#     Tref = 110. + 273.15  # [=] K.
#
#     # Define vectors with names of states
#     states = ['x']
#     nd = len(states)
#     xd = SX.sym('xd', nd)
#     for i in range(nd):
#         globals()[states[i]] = xd[i]
#
#     # Define vectors with names of algebraic variables
#     algebraics = ['Xa', 'Xb', 'Xc', 'Xe', 'Xp', 'Xg']
#     na = len(algebraics)
#     xa = SX.sym('xa', na)
#     for i in range(na):
#         globals()[algebraics[i]] = xa[i]
#
#     inputs = ['Fb', 'Tr']
#     nu = len(inputs)
#     u = SX.sym("u", nu)
#     for i in range(nu):
#         globals()[inputs[i]] = u[i]
#
#     # Reparametrization
#     k1 = 1.6599e6 * np.exp(-6666.7 / (Tr + 273.15))
#     k2 = 7.2117e8 * np.exp(-8333.3 / (Tr + 273.15))
#     k3 = 2.6745e12 * np.exp(-11111. / (Tr + 273.15))
#
#     # reaction rate
#     Fr = Fa + Fb
#     r1 = k1 * Xa * Xb * Mt
#     r2 = k2 * Xb * Xc * Mt
#     r3 = k3 * Xc * Xp * Mt
#
#     # residual for x
#     x_res = np.zeros((6, 1))
#     x_res[0, 0] = (Fa - r1 - Fr * Xa) / Mt
#     x_res[1, 0] = (Fb - r1 - r2 - Fr * Xb) / Mt
#     x_res[2, 0] = (+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt
#     x_res[3, 0] = (+ 2 * r2 - Fr * Xe) / Mt
#     x_res[4, 0] = (+   r2 - 0.5 * r3 - Fr * Xp) / Mt
#     x_res[5, 0] = (+ 1.5 * r3 - Fr * Xg) / Mt
#     # Define vectors with banes of input variables
#
#     ODEeq = [0 * x]
#
#     # Declare algebraic equations
#     Aeq = []
#
#     Aeq += [(Fa - r1 - Fr * Xa) / Mt]
#     Aeq += [(Fb - r1 - r2 - Fr * Xb) / Mt]
#     Aeq += [(+ 2 * r1 - 2 * r2 - r3 - Fr * Xc) / Mt]
#     Aeq += [(+ 2 * r2 - Fr * Xe) / Mt]
#     Aeq += [(+   r2 - 0.5 * r3 - Fr * Xp) / Mt]
#     Aeq += [(+ 1.5 * r3 - Fr * Xg) / Mt]
#
#     return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs
#
#
# def integrator_system():
#     """
#     This function constructs the integrator to be suitable with casadi environment, for the equations of the model
#     and the objective function with variable time step.
#     inputs: NaN
#     outputs: F: Function([x, u, dt]--> [xf, obj])
#     """
#
#     xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = DAE_system()
#     VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
#     solver = rootfinder('solver', 'newton', VV)
#
#     return solver
#
#
# def WO_obj_sys_ca(u):
#     solver = integrator_system()
#     x = solver(np.zeros(6), u)
#     Fb = u[0]
#     Tr = u[1]
#     Fa = 1.8275
#     Fr = Fa + Fb
#
#     obj = -(1043.38 * x[4] * Fr +
#             20.92 * x[3] * Fr -
#             79.23 * Fa -
#             118.34 * Fb)
#
#     return obj
#
#
# def WO_con1_sys_ca(u):
#     solver = integrator_system()
#     x = solver(np.zeros(6), u)
#     pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(1., 1)
#
#     return -pcon1
#
#
# def WO_con2_sys_ca(u):
#     solver = integrator_system()
#     x = solver(np.zeros(6), u)
#     pcon2 = x[5] - 0.08  # + 5e-4*np.random.normal(1., 1)
#
#     return -pcon2
#
options = {'disp': False, 'maxiter': 10000}  # solver options
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


# --- residual function for model opt --- #
def WO_nonlinear_f_model_opt(x, u_):
    Fb = u_[0]
    Tr = u_[1]

    # states of the system
    Xa = x[0]  # Mass fraction
    Xb = x[1]  # Mass fraction
    Xe = x[2]  # Mass fraction
    Xp = x[3]  # Mass fraction
    Xg = x[4]  # Mass fraction

    # Reparametrization
    k1 = np.exp(phi1) * np.exp((Tref / (Tr + 273.15) - 1) * psi1)
    k2 = np.exp(phi2) * np.exp((Tref / (Tr + 273.15) - 1) * psi2)

    # reaction rate
    Fr = Fa + Fb
    r1 = k1 * Xa * Xb * Xb * Mt
    r2 = k2 * Xa * Xb * Xp * Mt

    # residual for x
    x_res = np.zeros((5, 1))
    x_res[0, 0] = Fa - r1 - r2 - Fr * Xa
    x_res[1, 0] = Fb - 2 * r1 - r2 - Fr * Xb
    x_res[2, 0] = + 2 * r1 - Fr * Xe
    x_res[3, 0] = +   r1 - r2 - Fr * Xp
    x_res[4, 0] = + 3 * r2 - Fr * Xg

    return np.sum(x_res ** 2)


# --- residual function for model --- #
def WO_nonlinear_f_model(u_, x):
    Fb = u_[0]
    Tr = u_[1]

    # states of the system
    Xa = x[0]  # Mass fraction
    Xb = x[1]  # Mass fraction
    Xe = x[2]  # Mass fraction
    Xp = x[3]  # Mass fraction
    Xg = x[4]  # Mass fraction

    # Reparametrization
    k1 = np.exp(phi1) * np.exp((Tref / (Tr + 273.15) - 1) * psi1)
    k2 = np.exp(phi2) * np.exp((Tref / (Tr + 273.15) - 1) * psi2)

    # reaction rate
    Fr = Fa + Fb
    r1 = k1 * Xa * Xb * Xb * Mt
    r2 = k2 * Xa * Xb * Xp * Mt

    # residual for x
    x_res = np.zeros((5, 1))
    x_res[0, 0] = Fa - r1 - r2 - Fr * Xa
    x_res[1, 0] = Fb - 2 * r1 - r2 - Fr * Xb
    x_res[2, 0] = + 2 * r1 - Fr * Xe
    x_res[3, 0] = +   r1 - r2 - Fr * Xp
    x_res[4, 0] = + 3 * r2 - Fr * Xg

    return x_res


# --- WO model objective --- #
def WO_Model_obj(u):
    x_guess = np.ones((5, 1)) * 0.2
    WO_f_model = functools.partial(WO_nonlinear_f_model, u)
    x_solved = broyden1(WO_f_model, x_guess, f_tol=1e-12)

    # definitions
    Fa = 1.8275
    Fb = u[0]
    Fr = Fa + Fb

    # calculating objective
    obj = -(1043.38 * x_solved[3, 0] * Fr +
            20.92 * x_solved[2, 0] * Fr -
            79.23 * Fa -
            118.34 * Fb)

    return obj


# --- WO model con1 --- #
def WO_Model_con1(u):
    x_guess = np.ones((5, 1)) * 0.2
    WO_f_model = functools.partial(WO_nonlinear_f_model, u)
    x_solved = broyden1(WO_f_model, x_guess, f_tol=1e-12)

    # calculating con1
    con1 = x_solved[0, 0] - 0.12

    return -con1


# --- WO model con1 opt --- #
def WO_Model_con1_opt(u):
    x_guess = np.ones((5, 1)) * 0.2

    res = minimize(WO_nonlinear_f_model_opt, x_guess, args=(u),
                   method='BFGS', options=options, tol=1e-12)
    x_solved = res.x

    # calculating con1
    con1 = x_solved[0] - 0.12

    return -con1


# --- WO model con2 --- #
def WO_Model_con2(u):
    x_guess = np.ones((5, 1)) * 0.2
    WO_f_model = functools.partial(WO_nonlinear_f_model, u)
    x_solved = broyden1(WO_f_model, x_guess, f_tol=1e-12)

    # calculating con1
    con2 = x_solved[4, 0] - 0.08

    return -con2


# --- WO model con2 opt --- #
def WO_Model_con2_opt(u):
    x_guess = np.ones((5, 1)) * 0.2
    res = minimize(WO_nonlinear_f_model_opt, x_guess, args=(u),
                   method='BFGS', options=options, tol=1e-12)
    x_solved = res.x

    # calculating con1
    con2 = x_solved[4] - 0.08

    return -con2
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


def DAE_system():
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


def integrator_system():
    """
    This function constructs the integrator to be suitable with casadi environment, for the equations of the model
    and the objective function with variable time step.
    inputs: NaN
    outputs: F: Function([x, u, dt]--> [xf, obj])
    """

    xd, xa, u, ODEeq, Aeq, states, algebraics, inputs = DAE_system()
    VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
    solver = rootfinder('solver', 'newton', VV)

    return solver


def WO_obj_sys_ca(u):
    solver = integrator_system()
    x = solver(np.zeros(6), u)
    Fb = u[0]
    Tr = u[1]
    Fa = 1.8275
    Fr = Fa + Fb

    obj = -(1043.38 * x[4] * Fr +
            20.92 * x[3] * Fr -
            79.23 * Fa -
            118.34 * Fb)

    return obj


def WO_con1_sys_ca(u):
    solver = integrator_system()
    x = solver(np.zeros(6), u)
    pcon1 = x[0] - 0.12  # + 5e-4*np.random.normal(1., 1)

    return -pcon1.toarray()[0]


def WO_con2_sys_ca(u):
    solver = integrator_system()
    x = solver(np.zeros(6), u)
    pcon2 = x[5] - 0.08  # + 5e-4*np.random.normal(1., 1)

    return -pcon2.toarray()[0]


class Comp_system:
    # Parameters


    def __init__(self, a, b):
        # self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.states, self.algebraics, self.inputs = self.DAE_system()

        self.N_c  = np.shape(a)[0] # Number of compressors
        self.eval = []
        self.R    = 8.314e-3  # Power in MW
        self.MW   = 16.04e0
        self.Zin  = 0.90e0
        self.Tin  = 293e0
        self.nv   = 1.27e0
        self.Pin  = 1.05e5
        self.Pout = 1.55e5
        self.kin  = 2.2627e0
        self.kout = 0.9410e0
        self.krec = 0.4238e0
        self.s0   = -6.64e-1
        self.s1   = 3.81e-2
        self.c0   = 3.41e-1
        self.c1   = 9.66e-3
        for i in range(self.N_c):
            self.eval += [self.integrator_system(a[i], b[i])]
        print(self.map_pi(a[0], 80, 39))

    def map_pi(self, a, Mc, W):
        return a[0] + a[1] * W + a[2] * Mc + a[3] * Mc * W + a[4] * (W)**2 + a[5] * (Mc)**2
#a[0] + a[1] * Mc + a[2] * W + a[3] * Mc * W + a[4] * (Mc)**2 + a[5] * (W)**2
    def map_pi1(self, a, Mc, W):
        return a[0] + a[1] * Mc + a[2] * W + a[3] * Mc * W + a[4] * (Mc)**2 + a[5] * (W)**2
    def map_eta(self, b, Mc, PI):
        return b[0] + b[1] * Mc + b[2] * PI + b[3] * Mc * PI + b[4] * (Mc)**2 + b[5] * (PI)**2


    def DAE_system(self, a, b):
        # Define vectors with names of states
        states = ['x']
        nd = len(states)
        xd = SX.sym('xd', nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]
        # Define vectors with names of algebraic variables

        algebraics = ['Min', 'Mout', 'Mrec', 'Mc',
                      'Ps', 'Pd', 'PI', 'Ep', 'Yp', 'P']

        na = len(algebraics)
        xa = SX.sym('xa', na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        inputs = ['W_norm', 'Vrec_norm']
        nu = len(inputs)
        u = SX.sym("u", nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]
        # Reparametrization
        # Define vectors with banes of input variables

        ODEeq = [0 * x]

        # Declare algebraic equations

        W_min = 1e1
        W_max = 1e2
        Vrec_min = 0e0
        Vrec_max = 1.

        Vrec =Vrec_norm * (Vrec_max - Vrec_min) + Vrec_min
        W =  W_norm * (W_max - W_min) + W_min

        Aeq = []

        Aeq += [(Min /self.kin)**2 - ((self.Pin - Ps))]

        Aeq += [(Mout/ self.kout)**2 - ((Pd - self.Pout))]

        Aeq += [(Mrec/ self.krec)**2 - (Vrec)**2 * ((Pd - Ps))]

        Aeq += [Mout - Min]

        Aeq += [Mc - Min - Mrec]

        Aeq += [Pd -  PI*Ps]

        Aeq += [PI - self.map_pi1(a, Mc, W)]

        Aeq += [Ep - self.map_eta(b, Mc, PI)]

        Aeq += [Yp - ((self.Zin * self.R * self.Tin) / self.MW) * (self.nv / (self.nv - 1)) * (
                PI**((self.nv - 1) / self.nv) - 1)]


        Aeq += [P * Ep - Yp * Mc]



        geq = [*Aeq]
        geq += [self.Pin - Ps]
        geq += [Pd - self.Pout]
        geq += [Pd - Ps]

        return xd, xa, u, ODEeq, Aeq, states, algebraics, inputs, geq

    def integrator_system(self, a, b):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
        inputs: NaN
        outputs: F: Function([x, u, dt]--> [xf, obj])
        """
        xd, xa, u, ODEeq, Aeq, states, algebraics, inputs, geq = self.DAE_system(a, b)
        # lbg = np.zeros(np.shape(geq)[0])-1e-7
        # ubg = np.hstack((np.zeros(np.shape(Aeq)[0])+1e-7,[inf]*3))
        # lbw = np.zeros(np.shape(Aeq)[0])
        # ubw = [inf]*np.shape(Aeq)[0]
        # mse = sum1(vertcat(*Aeq)**2)/10
        # problem = {'f': mse, 'x': xa, 'g': vertcat(*geq)}
        # solver = nlpsol('solver', 'ipopt', problem, {
        #     "ipopt.print_level": 5})  # , {"ipopt.hessian_approximation":"limited-memory"})#, {"ipopt.tol": 1e-10, "ipopt.print_level": 0})#, {"ipopt.hessian_approximation":"limited-memory"})
        # sol = solver(x0=lbw, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        # x1 = sol['x']
        # print(x1)
        # self.x1 =x1
        VV = Function('vfcn', [xa, u], [vertcat(*Aeq)], ['w0', 'u'], ['w'])
        opt = dict(print_iteration=0,error_on_fail=False, max_iter=1e5)
        solver = rootfinder('solver', 'newton', VV, opt)


        return solver

    def obj_sys_ca(self, u):
        obj = 0
        xx = []
        for i in range(self.N_c):
            x1 = self.eval[i](np.array([50., 50., 50., 10., 1.2e5, 2.e5 , 1.0e0, 1.0e-2, 0.1, 0.1]), u[i*2:2*i+2])#
            obj += x1[-1]
            xx  += [x1]
        self.xx = xx


        return obj

    def obj_sys_ca_noise_less(self, u):
        obj = 0
        xx = []
        for i in range(self.N_c):
            x1 = self.eval[i](np.array([250, 250, 250, 250, 1.0e5, 1.0e5]), u[i*2:2*i+2])#, 1.0e0, 1.0e-2, 0.1, 0.1
            obj += x1[-1]
            xx  += [x1]
        self.xx = xx
        obj = -obj

        return obj

    def con11_sys_ca(self, u):
        u1 = u[:2]
        x1 = self.eval[0](np.array([50., 50., 50., 10., 1.2e5, 2.e5 , 1.0e0, 1.0e-2, 0.1, 0.1]), u1)  #

        g_1 = x1[6] - self.s0 - self.s1 * x1[3]
        # x1[6] - self.c0 - self.c1 * x1[3]
        pcon1 = g_1#- 0.12 + 5e-4 * np.random.normal(0., 1)

        return -pcon1.toarray()[0]

    def con12_sys_ca(self, u):
        u1 = u[:2]
        x1 = self.eval[0](np.array([50, 50, 50, 50, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u1)
        # g_1 = x1[6] - self.s0 - self.s1 * x1[3]
        g_1 = x1[6] - self.c0 - self.c1 * x1[3]
        pcon1 = g_1#- 0.12 + 5e-4 * np.random.normal(0., 1)

        return pcon1.toarray()[0]

    def con21_sys_ca(self, u):
        u2 = u[2:4]
        x1 = self.eval[1](np.array([50, 50, 50, 50, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u2)
        g_1 = x1[6] - self.s0 - self.s1 * x1[3]
        # x1[6] - self.c0 - self.c1 * x1[3]
        pcon1 = g_1#- 0.12 + 5e-4 * np.random.normal(0., 1)

        return -pcon1.toarray()[0]

    def con22_sys_ca(self, u):
        u2 = u[2:4]
        x1 = self.eval[1](np.array([50, 50, 50, 50, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u2)
        # g_1 = x1[6] - self.s0 - self.s1 * x1[3]
        g_1 = x1[6] - self.c0 - self.c1 * x1[3]
        pcon1 = g_1#- 0.12 + 5e-4 * np.random.normal(0., 1)

        return pcon1.toarray()[0]

    def equality_sys_ca(self, ref, u):
        con = 0

        for i in range(self.N_c):
            x1 = self.eval[i](np.array([250, 250, 250, 250, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u[i*2:2*i+2])
            con += x1[1]

        con = con - ref
        pcon = con#- 0.12 + 5e-4 * np.random.normal(0., 1)

        return -pcon.toarray()[0]

    def equality_right_sys_ca(self, ref, u):
        obj = 0
        xx = []
        for i in range(self.N_c):
            x1 = self.eval[i](np.array([250, 250, 250, 250, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u[i*2:2*i+2])
            obj += x1
            xx  += [x1]
        self.xx = xx
        obj = -obj

        return -pcon2.toarray()[0]


    def compliment_sys_ca(self, u):
        x1 = self.eval1(np.array([250, 250, 250, 250, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u)
        x2 = self.eval2(np.array([250, 250, 250, 250, 1.0e5, 1.0e5, 1.0e0, 1.0e-2, 0.1, 0.1]), u)
        self.x1 = x1
        self.x2 = x2
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
