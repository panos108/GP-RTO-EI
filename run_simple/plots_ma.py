s = np.array(pd.read_csv('FDK05new3.csv'))
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
n = 30
ni = 20

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

useful = s[:43 * 30, :4]
reshaped_use = useful.reshape((43, 30, 4))
for im in range(30):
    ax.plot(s[41 * im:41 * im + 41, 1],
            s[41 * im:41 * im + 41, 2],
            color='#AA4339', alpha=.5, marker='o', linestyle='-')
ax.plot(reshaped_use[0, 0, 1], reshaped_use[0, 0, 1], 'k*')

# ax.plot(s[:,1],
#         s[:,2],
#         color='#255E69', alpha=.5, marker='h', linestyle='None')

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
for im in range(30):
    ax.plot(s[41 * im + 41 - 5:41 * im + 41, 1],
            s[41 * im + 41 - 5:41 * im + 41, 2], marker=6, color='#7B9F35')

# plt.axis([4.,7., 70.,100.])
plt.plot(0.36730946, -0.39393939, marker='*', color='#255E69')
plt.xlabel(r'$u_1$')
plt.ylabel(r'$u_2$')
plt.tick_params(right=True, top=True, left=True, bottom=True)
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.axis([-1., 1.5, -1, 1])
plt.tight_layout()
plt.savefig('Contour_prob_ma_k_05.png', dpi=400)




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
objective_02 = compute_obj_simple_ma('FD_K02_new3.csv')
objective_05 = compute_obj_simple_ma('FDK05new3.csv')
objective_08 = compute_obj_simple_ma('FD_K08_new3.csv')

data = [  # obj_no_prior_with_exploration_ei[-1],
    objective_02[-1],
    objective_05[-1],
    # obj_no_prior_with_exploration_ucb[-1],
    objective_08[-1],]

ni = 41
color = ['AA6C39', 'AA3939', '226666', '7B9F35']
label = ['K=0.2', 'K=0.5', 'K=0.8']
for i, obj_ in reversed(list((enumerate(data)))):
    obj_mean = obj_.mean(axis=0)
    obj_max = obj_.max(axis=0)
    obj_min = obj_.min(axis=0)
    plt.plot(np.linspace(1, ni, ni), obj_mean,
             alpha=1., color='#' + color[i], label=label[i])
    plt.fill_between(np.linspace(1, ni, ni), np.quantile(obj_, 0.05, axis=0)
                     , np.quantile(obj_, 0.95, axis=0),
                     alpha=0.2, color='#' + color[i])
plt.plot(np.linspace(1, ni, ni), [0.145] * ni, 'k--', label='Real Optimum')
plt.xlabel('RTO-iter')
plt.ylabel('Objective')
plt.xlim(1, 20)
plt.legend()
plt.tick_params(right=True, top=True, left=True, bottom=True)
plt.tick_params(axis="y", direction="in")
plt.tick_params(axis="x", direction="in")
plt.tight_layout()
plt.savefig('objs_20.png', dpi=400)
plt.close()

