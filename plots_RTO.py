import matplotlib.pyplot as plt
import numpy as np
import pickle
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
def Plot(path):
    csfont = {'fontname': 'Times New Roman'}

    #plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "Times New Roman"

    #grid_shape = (1, 2)
    # fig = plt.figure()
    ft = int(20)
    font = {'size': ft}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}
    plt.rcParams.update(params)


    #
    X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc = \
         pickle.load(open(path+'.p','rb'))
    #
    n = 20
    ni = 20
    for i in range(n):
            plt.plot(np.linspace(1,ni,ni), np.array(TR_l_mc[i])[:ni].T,
                     color='#255E69',alpha = 1.)
    plt.xlabel('RTO-iter')
    plt.ylabel('Trust region radius')
    plt.tick_params(right= True,top= True,left= True, bottom= True)
    plt.tick_params(axis="y",direction="in")
    plt.tick_params(axis="x",direction="in")
    plt.tight_layout()
    plt.savefig('figs_WO/'+path + 'TR_prob.png',dpi=400)
    plt.close()




    model = WO_model()
    plant = WO_system()

    obj_model = model.WO_obj_ca
    cons_model = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system = plant.WO_obj_sys_ca_noise_less
    cons_system = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]

    n_iter = 40
    bounds = [[4., 7.], [70., 100.]]
    Xtrain = np.array([[5.7, 74.], [6.35, 74.9], [6.6, 75.], [6.75, 79.]])  # U0
    samples_number = Xtrain.shape[0]
    data = ['data0', Xtrain]
    u0 = np.array([6.9, 83])

    Delta0 = 0.25
    Delta_max = 0.7;
    eta0 = 0.2;
    eta1 = 0.8;
    gamma_red = 0.8;
    gamma_incr = 1.2;
    TR_scaling_ = False
    TR_curvature_ = False
    inner_TR_ = False

    # Plot a sin curve using the x and y axes.
    n_points = 40
    u_1 = np.linspace(4., 7., n_points)
    u_2 = np.linspace(70., 100., n_points)

    # --- plotting functions --- #
    plant_f_grid = np.zeros((n_points, n_points))  # f_copy
    for u1i in range(len(u_1)):
        for u2i in range(len(u_2)):
            try:
                plant_f_grid[u1i, u2i] = obj_system(np.array([u_1[u1i], u_2[u2i]]))
            except:
                print('point ', (u1i, u2i), ' failed')

    plant_f_grid = plant_f_grid.T
    # normalizing plot
    np_bounds = np.array(bounds)
    np_diff_bounds = np_bounds[:, 1] - np_bounds[:, 0]
    u_1 = np.linspace(4., 7., n_points)
    u_2 = np.linspace(70., 100., n_points)
    # --- constraints mapping manually set (lazyness) --- #
    g1 = [[4.0000, 4.1000, 4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000, 4.9000, \
           5.0000, 5.1000, 5.2000, 5.3000, 5.4000, 5.5000, 5.6000, 5.7000, 5.8000, 5.9000, 6.0000, \
           6.1000, 6.2000, 6.3000, 6.4000, 6.5000, 6.6000, 6.7000, 6.8000, 6.9000, 7.0000], \
          [83.8649, 82.9378, 82.0562, 81.2152, 80.4109, 79.6398, 78.8988, 78.1846, 77.4959, \
           76.8299, 76.1849, 75.5589, 74.9507, 74.3586, 73.7814, 73.2178, 72.6674, 72.1277, \
           71.5987, 71.0793, 70.5673, 70.0665, 69.5730, 69.0863, 68.6056, 68.1305, 67.6604, \
           67.1943, 66.7337, 66.2754, 65.8211]]
    g2 = [[4.0000, 4.1000, 4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000, 4.9000, \
           5.0000, 5.1000, 5.2000, 5.3000, 5.4000, 5.5000, 5.6000, 5.7000, 5.8000, 5.9000, 6.0000, \
           6.1000, 6.2000, 6.3000, 6.4000, 6.5000, 6.6000, 6.7000, 6.8000, 6.9000, 7.0000], \
          [78.0170, 78.6381, 79.2771, 79.9189, 80.5635, 81.2110, 81.8619, 82.5181, 83.1687, \
           83.8277, 84.4897, 85.1547, 85.8229, 86.4940, 87.1413, 87.8557, 88.5343, 89.2170, \
           89.9037, 90.5943, 91.2896, 91.9891, 92.6930, 93.4017, 94.1155, 94.7974, 95.5153, \
           96.2377, 96.9647, 97.6964, 98.4330]]

    # Contour plot
    # f_copy = f.reshape((n_points,n_points),order='F')

    samples_number = 4
    fig, ax = plt.subplots()
    CS = ax.contour(u_1, u_2, plant_f_grid, 50)
    ax.plot(g1[0]  , g1[1]  , 'black', linewidth=3)
    ax.plot(g2[0]  , g2[1]  , 'black', linewidth=3)
    # for im in range(n):
    #     ax.plot(X_opt_mc[im][samples_number:, 0]  ,
    #             X_opt_mc[im][samples_number:, 1]  ,
    #             color='#AA4339',alpha = .5, marker='o')
    #     ax.plot(X_opt_mc[im][:samples_number, 0]  ,
    #             X_opt_mc[im][:samples_number, 1]  ,
    #             color='#255E69',alpha = .5, marker='o', linestyle='None')
    #     tr_bktrc = 0
    #     for i in range(X_opt_mc[im][samples_number:, :].shape[0]):
    #         if backtrack_1_mc[im][i] == False:
    #             x_pos = X_opt_mc[im][samples_number + i, 0]
    #             y_pos = X_opt_mc[im][samples_number + i, 1]
    #             # plt.text(x_pos, y_pos, str(i))
    #         if TR_scaling_:
    #             if TR_curvature_:
    #                 print('Not implemented')
    #                 # e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0], TR_l_[im][i][1],
    #                 #              facecolor='None', edgecolor='black', angle=TR_l_angle[i], linestyle='--', linewidth=1)
    #                 # ax.add_patch(e2)
    #             else:
    #                 e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0][0], TR_l_mc[im][i][1][0],
    #                              facecolor='None', edgecolor='black', angle=0, linestyle='--', linewidth=1)
    #                 ax.add_patch(e2)
    #         else:
    #             2
    #             # circle1 = plt.Circle((x_pos, y_pos), radius=TR_l_mc[im][i], color='black', fill=False, linestyle='--')
    #             # ax.add_artist(circle1)
    # for im in range(n):
    #     ax.plot(xnew_mc[im][-5:, 0]  , xnew_mc[im][-5:, 1]  , 'ro')


    for im in range(n):
        ax.plot(X_opt_mc[im][samples_number:, 0],
                X_opt_mc[im][samples_number:, 1],
                color='#AA4339', alpha=.5, marker='o')
        ax.plot(X_opt_mc[im][:samples_number, 0],
                X_opt_mc[im][:samples_number, 1],
                color='#255E69', alpha=.5, marker='h', linestyle='None')
        tr_bktrc = 0
        # for i in range(X_opt_mc[im][samples_number:, :].shape[0]):
        #     if backtrack_1_mc[im][i] == False:
        #         x_pos = X_opt_mc[im][samples_number + i, 0]
        #         y_pos = X_opt_mc[im][samples_number + i, 1]
        #         # plt.text(x_pos, y_pos, str(i))
        #     if TR_scaling_:
        #         if TR_curvature_:
        #             print('Not implemented')
        #             # e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0], TR_l_[im][i][1],
        #             #              facecolor='None', edgecolor='black', angle=TR_l_angle[i], linestyle='--', linewidth=1)
        #             # ax.add_patch(e2)
        #         else:
        #             e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0][0], TR_l_mc[im][i][1][0],
        #                          facecolor='None', edgecolor='black', angle=0, linestyle='--', linewidth=1)
        #             ax.add_patch(e2)
        #     else:
        #         2
        #         # circle1 = plt.Circle((x_pos, y_pos), radius=TR_l_mc[im][i], color='black', fill=False, linestyle='--')
        #         # ax.add_artist(circle1)
    for im in range(n):
        ax.plot(xnew_mc[im][-5:, 0], xnew_mc[im][-5:, 1], marker=6, color='#7B9F35')
    plt.plot(64.389, 80.49, marker='*', color='#255E69')


        # plt.axis([4.,7., 70.,100.])

    plt.xlabel(r'Mass Flowrate of B [kg s$^-1$]')
    plt.ylabel(r'Reactor Temperature [$^o$C]')
    plt.tick_params(right= True,top= True,left= True, bottom= True)
    plt.tick_params(axis="y",direction="in")
    plt.tick_params(axis="x",direction="in")
    plt.xlim(4,7)
    plt.ylim(70,100)
    plt.tight_layout()
    plt.savefig('figs_WO/'+path+'Contour_prob.png',dpi=400)
    plt.close()


    #------------------------------------------------#

    # obj_ = np.zeros([n, ni])
    # for i in range(n):
    #     obj_list = []
    #     for p_i in range(ni):
    #         if cons_system[0](np.array(X_opt_mc)[i, samples_number + p_i - 1, :]) > 0 or cons_system[0](
    #                 np.array(X_opt_mc)[i, samples_number + p_i - 1, :]) > 0:
    #             print(1)
    #             obj_[i, p_i] = (-obj_system(np.array(X_opt_mc)[i, samples_number + p_i, :]) + 200.)
    #         else:
    #             obj_[i, p_i] = (-obj_system(np.array(X_opt_mc)[i, samples_number + p_i, :]) + 200.)
    # obj_mean = obj_.mean(axis=0)
    # obj_max = obj_.max(axis=0)
    # obj_min = obj_.min(axis=0)
    #
    # plt.errorbar(np.linspace(1, ni, ni), obj_mean, yerr=[obj_mean - obj_min, obj_max - obj_mean],
    #              color='#255E69', alpha=1.)
    # #
    # # plt.plot(np.linspace(1, ni, ni), obj_max,
    # #              color='#255E69', alpha=1.)
    # #
    # # plt.plot(np.linspace(1, ni, ni), obj_min,
    # #              color='#255E69', alpha=1.)
    # # plt.plot(np.linspace(1, ni, ni), [obj_max.max()]*ni,
    # #              color='#255E69', alpha=1.)
    # plt.xlabel('RTO-iter')
    # plt.ylabel('Objective')
    # plt.tick_params(right=True, top=True, left=True, bottom=True)
    # plt.tick_params(axis="y", direction="in")
    # plt.tick_params(axis="x", direction="in")
    # plt.tight_layout()
    # plt.savefig(path + 'obj.png',dpi=400)
    # plt.close()
    # print('end')
    return print('end')


def Plot_simple(path):
    csfont = {'fontname': 'Times New Roman'}

    # plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "Times New Roman"

    # grid_shape = (1, 2)
    # fig = plt.figure()
    ft = int(20)
    font = {'size': ft}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}
    plt.rcParams.update(params)

    #
    X_opt_mc, y_opt_mc, TR_l_mc, xnew_mc, backtrack_1_mc = \
        pickle.load(open(path + '.p', 'rb'))
    #
    n = 30
    ni = 20
    for i in range(n):
        plt.plot(np.linspace(1, ni, ni), np.array(TR_l_mc[i])[:ni].T,
                 color='#255E69', alpha=1.)
    plt.xlabel('RTO-iter')
    plt.ylabel('Trust region radius')
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs/' + path + 'TR_prob.png', dpi=400)
    plt.close()

    n_points = 20
    x_1 = np.linspace(-1., 1.5, n_points)
    x_2 = np.linspace(-1., 1., n_points)
    x = [[x, y] for x in x_1 for y in x_2]
    x = np.array(x)
    x = x.T

    # --- plotting functions --- #

    # plotting objective
    def simplefunc_plot(x):
        return x[0, :] ** 2 + x[1, :] ** 2 + x[0, :] * x[1, :]
        # return (x[0,:]-1.)**2 + 5*(x[1,:]+1.)**2

    f = simplefunc_plot(x)
    # plotting constraint g1
    g11x2Simple_plot = [x ** 2 + 2. * x + 1. for x in x_2]

    # Contour plot
    f_copy = f.reshape((n_points, n_points), order='F')

    fig, ax = plt.subplots()
    CS = ax.contour(x_1, x_2, f_copy, 50)
    ax.plot(g11x2Simple_plot, x_2, 'black', linewidth=3)

    plt.axis([-1., 1.5, -1, 1])

    samples_number = 3
    for im in range(n):
        ax.plot(X_opt_mc[im][samples_number:, 0],
                X_opt_mc[im][samples_number:, 1],
                color='#AA4339', alpha=.5, marker='o')
        ax.plot(X_opt_mc[im][:samples_number, 0],
                X_opt_mc[im][:samples_number, 1],
                color='#255E69', alpha=.5, marker='h', linestyle='None')
        tr_bktrc = 0
        # for i in range(X_opt_mc[im][samples_number:, :].shape[0]):
        #     if backtrack_1_mc[im][i] == False:
        #         x_pos = X_opt_mc[im][samples_number + i, 0]
        #         y_pos = X_opt_mc[im][samples_number + i, 1]
        #         # plt.text(x_pos, y_pos, str(i))
        #     if TR_scaling_:
        #         if TR_curvature_:
        #             print('Not implemented')
        #             # e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0], TR_l_[im][i][1],
        #             #              facecolor='None', edgecolor='black', angle=TR_l_angle[i], linestyle='--', linewidth=1)
        #             # ax.add_patch(e2)
        #         else:
        #             e2 = Ellipse((x_pos, y_pos), TR_l_mc[im][i][0][0], TR_l_mc[im][i][1][0],
        #                          facecolor='None', edgecolor='black', angle=0, linestyle='--', linewidth=1)
        #             ax.add_patch(e2)
        #     else:
        #         2
        #         # circle1 = plt.Circle((x_pos, y_pos), radius=TR_l_mc[im][i], color='black', fill=False, linestyle='--')
        #         # ax.add_artist(circle1)
    for im in range(n):
        ax.plot(xnew_mc[im][-5:, 0], xnew_mc[im][-5:, 1], marker=6, color='#7B9F35')

        # plt.axis([4.,7., 70.,100.])
    plt.plot(0.36730946, -0.39393939, marker='*', color='#255E69')
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.axis([-1., 1.5, -1, 1])
    plt.tight_layout()
    plt.savefig('figs/' + path + 'Contour_prob.png', dpi=400)
    plt.close()

    return print('end')


def compute_obj_simple(path):
    obj_system = Benoit_System_noiseless
    cons_system = [con1_system_tight_noiseless]
    X_opt_mc, y_opt_mc, TR_l_mc, xnew_mc, backtrack_1_mc = \
        pickle.load(open(path + '.p', 'rb'))
    #
    n = 30
    ni = 20
    samples_number = 3
    obj_1 = np.zeros([n, ni])
    for i in range(n):

        for p_i in range(ni):
            if cons_system[0](np.array(X_opt_mc)[i, samples_number + p_i, :]) < 0:  # np.array(backtrack_1_mc)[i,p_i]:
                for j in reversed(range(p_i)):
                    if cons_system[0](np.array(X_opt_mc)[i, samples_number + j,
                                      :]) >= 0:  # and np.array(backtrack_1_mc)[i,p_i-1]==False:
                        obj_1[i, p_i] = obj_1[i, j]
                        break

            else:
                obj_1[i, p_i] = obj_system(np.array(X_opt_mc)[i, samples_number + p_i, :])

    obj_mean = obj_1.mean(axis=0)
    obj_max = obj_1.max(axis=0)
    obj_min = obj_1.min(axis=0)

    return obj_mean, obj_max, obj_min, obj_1

def compute_obj(path):
    plant = WO_system()

    obj_system     = plant.WO_obj_sys_ca_noise_less
    cons_system    = [plant.WO_con1_sys_ca_noise_less, plant.WO_con2_sys_ca_noise_less]


    X_opt_mc, y_opt_mc, TR_l_mc, xnew_mc, backtrack_1_mc = \
        pickle.load(open(path + '.p', 'rb'))
    #
    n = 30
    ni = 20
    samples_number = 4
    obj_1 = np.zeros([n, ni])
    for i in range(n):

        for p_i in range(ni):
            if cons_system[0](np.array(X_opt_mc)[i, samples_number + p_i, :]) < 0 or cons_system[1](np.array(X_opt_mc)[i, samples_number + p_i, :]) < 0:  # np.array(backtrack_1_mc)[i,p_i]:
                for j in reversed(range(p_i)):
                    if cons_system[0](np.array(X_opt_mc)[i, samples_number + j,
                                      :]) >= 0 and cons_system[1](np.array(X_opt_mc)[i, samples_number + j,
                                      :]):   # and np.array(backtrack_1_mc)[i,p_i-1]==False:
                        obj_1[i, p_i] = obj_1[i, j]
                        break

            else:
                obj_1[i, p_i] = -obj_system(np.array(X_opt_mc)[i, samples_number + p_i, :])+200

    obj_mean = obj_1.mean(axis=0)
    obj_max = obj_1.max(axis=0)
    obj_min = obj_1.min(axis=0)

    return obj_mean, obj_max, obj_min, obj_1

def plot_obj(obj):

    csfont = {'fontname': 'Times New Roman'}

    # plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "Times New Roman"

    # grid_shape = (1, 2)
    # fig = plt.figure()
    ft = int(20)
    font = {'size': ft}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}
    plt.rcParams.update(params)

    obj_no_prior_with_exploration_ei = obj('no_prior_with_exploration_ei')
    obj_with_prior_with_exploration_ei = obj('with_prior_with_exploration_ei_new2')#'with_prior_with_exploration_ei')
    # obj_with_prior_with_exploration_ei_noise  = obj('with_prior_with_exploration_ei_noise')
    # obj_no_prior_with_exploration_ei_noise    = obj('no_prior_with_exploration_ei_noise')
    obj_no_prior_with_exploration_ucb = obj('no_prior_with_exploration_ucb')
    obj_with_prior_with_exploration_ucb = obj('with_prior_with_exploration_ucb_new2')
    # obj_with_prior_with_exploration_ucb_noise = obj('with_prior_with_exploration_ucb_noise')
    # obj_no_prior_with_exploration_ucb_noise   = obj('no_prior_with_exploration_ucb_noise')
    # obj_no_prior_no_exploration = obj('no_prior_no_exploration')
    obj_with_prior_no_exploration = obj('with_prior_no_exploration')
    # obj_with_prior_no_exploration_noise       = obj('with_prior_no_exploration_noise')
    # obj_no_prior_no_exploration_noise         = obj('no_prior_no_exploration_noise')

    data = [  # obj_no_prior_with_exploration_ei[-1],
        obj_with_prior_with_exploration_ei[-1],
        # obj_no_prior_with_exploration_ucb[-1],
        obj_with_prior_with_exploration_ucb[-1]]#,
        # obj_no_prior_no_exploration[-1]]
        #obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['AA3939', '226666']#, '7B9F35']
    label = ['EI', 'LCB']#, 'No Exploration']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')
    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_WO/EXplore_no_explore_obj.png', dpi=400)
    plt.close()

    data = [obj_no_prior_with_exploration_ei[-1],
            obj_with_prior_with_exploration_ei[-1]]
    # obj_no_prior_with_exploration_ucb[-1],
    # obj_with_prior_with_exploration_ucb[-1],
    # obj_no_prior_no_exploration[-1]]
    # obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['AA3939', '226666', '7B9F35']
    label = ['No Prior', 'Prior']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_WO/EI_prior.png', dpi=400)
    plt.close()

    data = [  # obj_no_prior_with_exploration_ei[-1],
        # obj_with_prior_with_exploration_ei[-1]]
        obj_no_prior_with_exploration_ucb[-1],
        obj_with_prior_with_exploration_ucb[-1]]
    # obj_no_prior_no_exploration[-1]]
    # obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['AA3939', '226666', '7B9F35']
    label = ['No Prior', 'Prior']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])

        print(obj_min.min())
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_WO/UCB_prior.png', dpi=400)
    plt.close()

    data = [  # obj_no_prior_with_exploration_ei[-1],
        # obj_with_prior_with_exploration_ei[-1]]
        # obj_no_prior_with_exploration_ucb[-1],
        # obj_with_prior_with_exploration_ucb[-1]]
        # obj_no_prior_no_exploration[-1]]
        obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['7B9F35']
    label = ['No Exploration']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])

        print(obj_min.min())
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_WO/noexplore_prior.png', dpi=400)
    plt.close()
    return print(1)


def plot_obj_noise(obj):
    csfont = {'fontname': 'Times New Roman'}

    # plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "Times New Roman"

    # grid_shape = (1, 2)
    # fig = plt.figure()
    ft = int(20)
    font = {'size': ft}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    params = {'legend.fontsize': 15,
              'legend.handlelength': 2}
    plt.rcParams.update(params)

    # obj_no_prior_with_exploration_ei = obj('no_prior_with_exploration_ei')
    # obj_with_prior_with_exploration_ei = obj('with_prior_with_exploration_ei')
    obj_with_prior_with_exploration_ei_noise = obj('with_prior_with_exploration_ei_noise')
    obj_no_prior_with_exploration_ei_noise = obj('no_prior_with_exploration_ei_noise')
    # obj_no_prior_with_exploration_ucb = obj('no_prior_with_exploration_ucb')
    # obj_with_prior_with_exploration_ucb = obj('with_prior_with_exploration_ucb')
    obj_with_prior_with_exploration_ucb_noise = obj('with_prior_with_exploration_ucb_noise')
    obj_no_prior_with_exploration_ucb_noise = obj('no_prior_with_exploration_ucb_noise')
    # obj_no_prior_no_exploration = obj('no_prior_no_exploration')
    # obj_with_prior_no_exploration = obj('with_prior_no_exploration')
    obj_with_prior_no_exploration_noise = obj('with_prior_no_exploration_noise')
    obj_no_prior_no_exploration_noise = obj('no_prior_no_exploration_noise')

    data = [  # obj_no_prior_with_exploration_ei[-1],
        obj_with_prior_with_exploration_ei_noise[-1],
        # obj_no_prior_with_exploration_ucb[-1],
        obj_with_prior_with_exploration_ucb_noise[-1],
        # obj_no_prior_no_exploration[-1]]
        obj_with_prior_no_exploration_noise[-1]]
    ni = 20
    color = ['AA3939', '226666', '7B9F35']
    label = ['EI', 'LCB', 'No Exploration']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')
    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_noise_WO/EXplore_no_explore_obj.png', dpi=400)
    plt.close()

    data = [obj_no_prior_with_exploration_ei_noise[-1],
            obj_with_prior_with_exploration_ei_noise[-1]]
    # obj_no_prior_with_exploration_ucb[-1],
    # obj_with_prior_with_exploration_ucb[-1],
    # obj_no_prior_no_exploration[-1]]
    # obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['AA3939', '226666', '7B9F35']
    label = ['No Prior', 'Prior']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_noise_WO/EI_prior.png', dpi=400)
    plt.close()

    data = [  # obj_no_prior_with_exploration_ei[-1],
        # obj_with_prior_with_exploration_ei[-1]]
        obj_no_prior_with_exploration_ucb_noise[-1],
        obj_with_prior_with_exploration_ucb_noise[-1]]
    # obj_no_prior_no_exploration[-1]]
    # obj_with_prior_no_exploration[-1]]
    ni = 20
    color = ['AA3939', '226666', '7B9F35']
    label = ['No Prior', 'Prior']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])

        print(obj_min.min())
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_noise_WO/UCB_prior.png', dpi=400)
    plt.close()

    data = [  # obj_no_prior_with_exploration_ei[-1],
        # obj_with_prior_with_exploration_ei[-1]]
        # obj_no_prior_with_exploration_ucb[-1],
        # obj_with_prior_with_exploration_ucb[-1]]
        # obj_no_prior_no_exploration[-1]]
        obj_with_prior_no_exploration_noise[-1]]
    ni = 20
    color = ['7B9F35']
    label = ['No Exploration']
    for i, obj_ in reversed(list((enumerate(data)))):
        obj_mean = obj_.mean(axis=0)
        obj_max = obj_.max(axis=0)
        obj_min = obj_.min(axis=0)
        plt.plot(np.linspace(1, ni, ni), obj_mean,
                 alpha=1., color='#' + color[i], label=label[i])
        plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                         , np.quantile(obj_, 0.95, axis=0),
                         alpha=0.2, color='#' + color[i])

        print(obj_min.min())
    plt.plot(np.linspace(1, ni, ni), [275.811] * ni, 'k--', label='Real Optimum')

    plt.xlabel('RTO-iter')
    plt.ylabel('Objective')
    plt.xlim(1, ni)
    plt.legend()
    plt.tick_params(right=True, top=True, left=True, bottom=True)
    plt.tick_params(axis="y", direction="in")
    plt.tick_params(axis="x", direction="in")
    plt.tight_layout()
    plt.savefig('figs_noise_WO/noexplore_prior.png', dpi=400)
    plt.close()
    return print(1)
