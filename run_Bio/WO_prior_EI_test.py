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
from run_Bio.samples_eval_gen import samples_generation
from casadi import *

from sub_uts.systems2 import *
from sub_uts.utilities_4 import *
# from plots_RTO import compute_obj, plot_obj, plot_obj_noise

import pickle
# from plots_RTO import Plot
#----------1) EI-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
if not(os.path.exists('figs_WO')):
    os.mkdir('figs_WO')
if not(os.path.exists('figs_noise_WO')):
    os.mkdir('figs_noise_WO')

#------------------------------------------------------------------
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []
plant = Bio_system(nk=2)

obj_system = plant.bio_obj_ca
cons_system = []  # l.WO_obj_ca
for k in range(plant.nk):
    cons_system.append(functools.partial(plant.bio_con1_ca, k + 1))
    cons_system.append(functools.partial(plant.bio_con2_ca, k + 1))

X, start = samples_generation(cons_system, obj_system, plant.nk*plant.nu)

for i in range(8):



    plant = Bio_system(nk=2)
    model = Bio_model(nk=2, empty=False)#empy=True)

    u = [0.]*plant.nk*plant.nu
    xf = plant.bio_obj_ca(u)
    x1 = plant.bio_con1_ca(1,u)
    x2 = plant.bio_con2_ca(1,u)
    functools.partial(plant.bio_con1_ca,1)
    obj_model  = model.bio_obj_ca_RK4#mode
    F = model.bio_model_ca()
    cons_model = []# l.WO_obj_ca
    for k in range(model.nk):
        cons_model.append(functools.partial(model.bio_con1_ca_RK4, k+1))
        cons_model.append(functools.partial(model.bio_con2_ca_RK4, k+1))



    # obj_model  = obj_empty
    # cons_model = []# l.WO_obj_ca
    # for k in range(model.nk):
    #     cons_model.append(con_empty)
    #     cons_model.append(con_empty)



    x = np.random.rand(model.nk*model.nu)
    x1= F(model.x0, x[:2])
    print(model.bio_con1_ca_f(x1), model.bio_con1_ca_RK4(1, x))

    n_iter         = 50
    bounds         = ([[0., 1.]] * model.nk*model.nu)#[[0.,1.],[0.,1.]]
    #X              = pickle.load(open('initial_data_bio_12_ca_new.p','rb'))
    Xtrain         = X[:model.nk*model.nu+1]#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
#
#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = X[start]#model.nk*model.nu+1]#np.array([*[0.6]*model.nk,*[0.8]*model.nk])#

    Delta0         = 0.25
    Delta_max      =5.; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]



    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=40,
                                    multi_hyper=10, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False, model=model)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei.p','wb'))


#-----------------Model---------

np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):



    plant = Bio_system()
    model = Bio_model()

    u = [0.]*plant.nk*plant.nu
    xf = plant.bio_obj_ca(u)
    x1 = plant.bio_con1_ca(1,u)
    x2 = plant.bio_con2_ca(1,u)
    functools.partial(plant.bio_con1_ca,1)
    obj_model  = model.bio_obj_ca_RK4#mode
    F = model.bio_model_ca()
    cons_model = []# l.WO_obj_ca
    for k in range(model.nk):
        cons_model.append(functools.partial(model.bio_con1_ca_RK4, k+1))
        cons_model.append(functools.partial(model.bio_con2_ca_RK4, k+1))



    # obj_model  = obj_empty
    # cons_model = []# l.WO_obj_ca
    # for k in range(model.nk):
    #     cons_model.append(con_empty)
    #     cons_model.append(con_empty)

    obj_system  = plant.bio_obj_ca
    cons_system = []# l.WO_obj_ca
    for k in range(model.nk):
        cons_system.append(functools.partial(plant.bio_con1_ca, k+1))
        cons_system.append(functools.partial(plant.bio_con2_ca, k+1))

    x = np.random.rand(24)
    print(F(model.x0, x[:2])[1] / 800 - 1, -model.bio_con1_ca_RK4(1, x))

    n_iter         = 8
    bounds         = ([[0., 1.]] * model.nk*model.nu)#[[0.,1.],[0.,1.]]
    X              = pickle.load(open('initial_data_bio_12_ca_new.p','rb'))
    Xtrain         = X[:model.nk*model.nu+1]#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
#
#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = X[18]#model.nk*model.nu+1]#np.array([*[0.6]*model.nk,*[0.8]*model.nk])#

    Delta0         = 0.5
    Delta_max      =5.; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=40,
                                    multi_hyper=10, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False, model=model)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei.p','wb'))




#-----------------------------------------------------------------------#
#----------2) EI-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ei.p','wb'))
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

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
#-----------------------------------------------------------------------#
#----------4) EI-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    plant = WO_system()

    obj_model      = obj_empty#model.WO_obj_ca
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ucb.p','wb'))
#-----------------------------------------------------------------------#
#----------2) UCB-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ucb.p','wb'))
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

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    plant = WO_system()

    obj_model      = obj_empty#model.WO_obj_ca
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_no_exploration.p','wb'))
#-----------------------------------------------------------------------#
#----------2) noexplorePRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration.p','wb'))
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

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

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
