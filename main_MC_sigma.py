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
from utilities import *
from systems import *

np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(20):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]


    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False

    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds), multi_opt=20,
                                    multi_hyper=10, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_)

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
import pickle
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_probabilistic.p','wb'))



# Plot a sin curve using the x and y axes.
n_points = 20
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
u_1 = np.linspace(4., 7., n_points) / np_diff_bounds[0]
u_2 = np.linspace(70., 100., n_points) / np_diff_bounds[1]
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


fig, ax = plt.subplots()
CS = ax.contour(u_1, u_2, plant_f_grid, 50)
ax.plot(g1[0] / np_diff_bounds[0], g1[1] / np_diff_bounds[1], 'black', linewidth=3)
ax.plot(g2[0] / np_diff_bounds[0], g2[1] / np_diff_bounds[1], 'black', linewidth=3)
ax.plot(X_opt[samples_number:, 0] / np_diff_bounds[0], X_opt[samples_number:, 1] / np_diff_bounds[1], 'ro')
ax.plot(X_opt[:samples_number, 0] / np_diff_bounds[0], X_opt[:samples_number, 1] / np_diff_bounds[1], '*')
tr_bktrc = 0
for i in range(X_opt[samples_number:, :].shape[0]):
    if backtrack_l[i] == False:
        x_pos = X_opt[samples_number + i, 0] / np_diff_bounds[0]
        y_pos = X_opt[samples_number + i, 1] / np_diff_bounds[1]
        # plt.text(x_pos, y_pos, str(i))
    if TR_scaling_:
        if TR_curvature_:
            e2 = Ellipse((x_pos, y_pos), TR_l[i][0], TR_l[i][1],
                         facecolor='None', edgecolor='black', angle=TR_l_angle[i], linestyle='--', linewidth=1)
            ax.add_patch(e2)
        else:
            e2 = Ellipse((x_pos, y_pos), TR_l[i][0][0], TR_l[i][1][0],
                         facecolor='None', edgecolor='black', angle=0, linestyle='--', linewidth=1)
            ax.add_patch(e2)
    else:
        circle1 = plt.Circle((x_pos, y_pos), radius=TR_l[i], color='black', fill=False, linestyle='--')
        ax.add_artist(circle1)

ax.plot(xnew[:, 0] / np_diff_bounds[0], xnew[:, 1] / np_diff_bounds[1], 'yo')
ax.set_title('Contour plot')
# plt.axis([4.,7., 70.,100.])
plt.show()

# compute all objective function values
obj_list = []
for p_i in range(X_opt[samples_number:, :].shape[0]):
    obj_list.append(obj_system(X_opt[samples_number + p_i, :]) + 100.)

fig, ax = plt.subplots()
ax.plot(obj_list)
plt.yscale('log')
ax.set_title('Objective function vs iterations')
plt.show()

# compute all objective function values
if not TR_scaling_:
    TR_list = []
    for p_i in range(len(TR_l)):
        TR_list.append(TR_l[p_i])

fig, ax = plt.subplots()
if not TR_scaling_:
    ax.plot(TR_list)
elif not TR_curvature_:
    ax.plot(np.linalg.norm(TR_l, axis=(1, 2)))
else:
    ax.plot(np.linalg.norm(TR_l, axis=(1)))
plt.yscale('log')
ax.set_title('TR vs iterations')
plt.show()