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
from sub_uts.utilities_2 import *
from sub_uts.systems import *
import pickle
from plots_RTO import plot_obj, Plot_simple, compute_obj_simple
if not(os.path.exists('figs')):
    os.mkdir('figs')
if not(os.path.exists('figs_noise')):
    os.mkdir('figs_noise')

print(2)
# plot_obj()
#----------2) EI-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
#
#
# from plots_RTO import compute_obj
# # obj_no_prior_with_exploration_ei          = compute_obj('no_prior_with_exploration_ei')
# # obj_with_prior_with_exploration_ei        = compute_obj('with_prior_with_exploration_ei')
# # obj_with_prior_with_exploration_ei_noise  = compute_obj('with_prior_with_exploration_ei_noise')
# # obj_no_prior_with_exploration_ei_noise    = compute_obj('no_prior_with_exploration_ei_noise')
# # obj_no_prior_with_exploration_ucb         = compute_obj('no_prior_with_exploration_ucb')
# # obj_with_prior_with_exploration_ucb       = compute_obj('with_prior_with_exploration_ucb')
# # obj_with_prior_with_exploration_ucb_noise = compute_obj('with_prior_with_exploration_ucb_noise')
# # obj_no_prior_with_exploration_ucb_noise   = compute_obj('no_prior_with_exploration_ucb_noise')
# # obj_no_prior_no_exploration               = compute_obj('no_prior_no_exploration')
# # obj_with_prior_no_exploration             = compute_obj('with_prior_no_exploration')
# # obj_with_prior_no_exploration_noise       = compute_obj('with_prior_no_exploration_noise')
# # obj_no_prior_no_exploration_noise         = compute_obj('no_prior_no_exploration_noise')
# #
#
#
#
# for i in range(30):
#
#     #model = WO_model()
# #plant = WO_system()
#
#     obj_model      = Benoit_Model
#     cons_model     = [con1_model]
#     obj_system     = Benoit_System
#     cons_system    = [con1_system_tight]
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ei.p','wb'))
#-----------------------------------------------------------------------#

#----------1) EI-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
# for i in range(30):
#
#     # #plant = WO_system()
#
#     obj_model      = obj_empty#Benoit_Model
#     cons_model     = [con_empty]
#     obj_system     = Benoit_System#Benoit_System
#     cons_system    = [con1_system_tight]#[con1_system_tight]
#
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei.p','wb'))
#-----------------------------------------------------------------------#

#----------3) EI-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    #model = WO_model()
#plant = WO_system()

    obj_model      = Benoit_Model
    cons_model     = [con1_model]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [ (1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ei_noise.p','wb'))
# #-----------------------------------------------------------------------#
# #----------4) EI-NO PRIOR-KNOWN NOISE----------#
# #---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    #plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [ (1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei_noise.p','wb'))
#-----------------------------------------------------------------------#


#-----------------------------UCB----------------------------------------------#
#----------1) UCB-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
# for i in range(30):
#
#     #plant = WO_system()
#
#     obj_model      = obj_empty#Benoit_Model
#     cons_model     = [con_empty]
#     obj_system     = Benoit_System
#     cons_system    = [con1_system_tight]
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ucb.p','wb'))
#-----------------------------------------------------------------------#
#----------2) UCB-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
# for i in range(30):
#
#     #model = WO_model()
# #plant = WO_system()
#
#     obj_model      = Benoit_Model
#     cons_model     = [con1_model]
#     obj_system     = Benoit_System
#     cons_system    = [con1_system_tight]
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ucb.p','wb'))
#-----------------------------------------------------------------------#
#----------3) UCB-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    #model = WO_model()
#plant = WO_system()

    obj_model      = Benoit_Model
    cons_model     = [con1_model]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [(1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ucb_noise.p','wb'))
#-----------------------------------------------------------------------#
#----------4) UCB-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    #plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [ (1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ucb_noise.p','wb'))
#-----------------------------------------------------------------------#


#-----------------------------No exploration----------------------------------------------#
#----------1) noexplore-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
# for i in range(30):
#
#     #plant = WO_system()
#
#     obj_model      = obj_empty#Benoit_Model
#     cons_model     = [con_empty]
#     obj_system     = Benoit_System
#     cons_system    = [con1_system_tight]
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_no_exploration.p','wb'))
#-----------------------------------------------------------------------#
#----------2) noexplorePRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
# np.random.seed(0)
# X_opt_mc = []
# y_opt_mc = []
# TR_l_mc = []
# xnew_mc = []
# backtrack_1_mc = []
#
# for i in range(30):
#
#     #model = WO_model()
# #plant = WO_system()
#
#     obj_model      = Benoit_Model
#     cons_model     = [con1_model]
#     obj_system     = Benoit_System
#     cons_system    = [con1_system_tight]
#
#
#
#
#     n_iter = 20
#     bounds = [[-.6, 1.5], [-1., 1.]]
#     Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
#     samples_number = Xtrain.shape[0]
#     data = ['data0', Xtrain]
#     u0 = np.array([1.1,-0.1])
#
#     Delta0         = 0.25
#     Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
#     TR_scaling_    = False
#     TR_curvature_  = False
#     inner_TR_      = False
#     noise = None#[0.5**2, 5e-8, 5e-8]
#
#
#     ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
#                                     Delta_max, eta0, eta1, gamma_red, gamma_incr,
#                                     n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
#                                     multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
#                                     store_data=True, inner_TR=inner_TR_, scale_inputs=False)
#
#     print('Episode: ',i)
#     if not TR_curvature_:
#         X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     else:
#         X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
#         backtrack_l                                       = [False, *backtrack_l]
#     X_opt_mc       += [X_opt]
#     y_opt_mc       += [y_opt]
#     TR_l_mc        += [TR_l]
#     xnew_mc        += [xnew]
#     backtrack_1_mc += [backtrack_l]
# print(2)
# pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration.p','wb'))
#-----------------------------------------------------------------------#
#----------3) no exploration-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    #model = WO_model()
#plant = WO_system()

    obj_model      = Benoit_Model
    cons_model     = [con1_model]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [ (1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration_noise.p','wb'))
#-----------------------------------------------------------------------#
#----------4) Noexplore-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
#----------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    #plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty]
    obj_system     = Benoit_System
    cons_system    = [con1_system_tight]




    n_iter = 20
    bounds = [[-.6, 1.5], [-1., 1.]]
    Xtrain         = np.array([[1.2, 0.],[1.4, 0.1],[1.3,-0.1]])
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([1.1,-0.1])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [ (1e-3)]*2


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_no_exploration_noise.p','wb'))
#-----------------------------------------------------------------------#



# Plot_simple('no_prior_with_exploration_ei')
# Plot_simple('with_prior_with_exploration_ei')
Plot_simple('with_prior_with_exploration_ei_noise')
Plot_simple('no_prior_with_exploration_ei_noise')
# Plot_simple('no_prior_with_exploration_ucb')
# Plot_simple('with_prior_with_exploration_ucb')
Plot_simple('with_prior_with_exploration_ucb_noise')
Plot_simple('no_prior_with_exploration_ucb_noise')
# Plot_simple('no_prior_no_exploration')
# Plot_simple('with_prior_no_exploration')
# Plot_simple('with_prior_no_exploration_noise')
Plot_simple('with_prior_no_exploration_noise')
Plot_simple('no_prior_no_exploration_noise')
from plots_RTO import compute_obj_simple
obj_no_prior_with_exploration_ei          = compute_obj_simple('no_prior_with_exploration_ei')
obj_with_prior_with_exploration_ei        = compute_obj_simple('with_prior_with_exploration_ei')
obj_with_prior_with_exploration_ei_noise  = compute_obj_simple('with_prior_with_exploration_ei_noise')
obj_no_prior_with_exploration_ei_noise    = compute_obj_simple('no_prior_with_exploration_ei_noise')
obj_no_prior_with_exploration_ucb         = compute_obj_simple('no_prior_with_exploration_ucb')
obj_with_prior_with_exploration_ucb       = compute_obj_simple('with_prior_with_exploration_ucb')
obj_with_prior_with_exploration_ucb_noise = compute_obj_simple('with_prior_with_exploration_ucb_noise')
obj_no_prior_with_exploration_ucb_noise   = compute_obj_simple('no_prior_with_exploration_ucb_noise')
obj_no_prior_no_exploration               = compute_obj_simple('no_prior_no_exploration')
obj_with_prior_no_exploration             = compute_obj_simple('with_prior_no_exploration')
obj_with_prior_no_exploration_noise       = compute_obj_simple('with_prior_no_exploration_noise')
obj_no_prior_no_exploration_noise         = compute_obj_simple('no_prior_no_exploration_noise')


