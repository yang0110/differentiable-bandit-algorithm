import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
import os 
#os.chdir('C:/DATA/Kaige_Research/Code/optimal_bandit/code/')
from linucb import LINUCB
from eliminator import ELI
from lse import LSE 
from lse_soft import LSE_soft
from lse_soft_online import LSE_soft_online
from lse_soft_base import LSE_soft_base
from lse_soft_v import LSE_soft_v
from linucb_soft import LinUCB_soft
from lints import LINTS
from linphe import LINPHE
from exp3 import EXP3
from giro import GIRO
from expucb_online import Expucb_online
from e_gready import E_greedy
from utils import *
path='../results/'
# np.random.seed(2018)

user_num=1
item_num=50
dimension=20
phase_num=11
iteration=2**phase_num
sigma=0.5 # noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.5
step_size_gamma=0.01
weight1=1
loop=1

beta=1.2
gamma=0
train_loops=150

beta_online=2.2

v=1
# a=0.25

exp3_gamma=3
eta=0.5


linucb_regret_matrix=np.zeros((loop, iteration))
lints_regret_matrix=np.zeros((loop, iteration))
giro_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
online_regret_matrix=np.zeros((loop, iteration))
offline_prob_matrix=np.zeros((loop, iteration))
online_prob_matrix=np.zeros((loop, iteration))
online_beta_matrix=np.zeros((loop, iteration))
exp3_regret_matrix=np.zeros((loop, iteration))
exp3_prob_matrix=np.zeros((loop, iteration))
e_regret_matrix=np.zeros((loop, iteration))

user_feature=np.random.normal(size=dimension)
user_feature=user_feature/np.linalg.norm(user_feature)
# train model
lse_soft_model=LSE_soft(dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

# lse_soft_regret_list_train, lse_soft_beta_list_train=lse_soft_model.train(train_loops, item_num)

# test data
# cov_matrix=np.identity(dimension)+np.random.normal(size=(dimension, dimension))
# cov_inv=np.linalg.pinv(cov_matrix)
# item_features=np.random.multivariate_normal(mean=np.zeros(dimension), cov=cov_inv, size=1*item_num)
item_features=np.random.normal(size=(item_num, dimension))
item_features=Normalizer().fit_transform(item_features)
# item_features=Normalizer().fit_transform(np.random.normal(size=(item_num, dimension)))
# item_features=np.random.normal(size=(item_num, dimension))
true_payoffs=np.dot(item_features, user_feature)
best_arm=np.argmax(true_payoffs)


for l in range(loop):
	linucb_model=LINUCB(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, beta)

	lints_model=LINTS(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, delta, sigma, v)

	online_model=Expucb_online(dimension, iteration, item_num, user_feature,item_features, true_payoffs, alpha, sigma, step_size_beta, weight1, beta_online)


	e_model=E_greedy(dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma)
	#####################

	linucb_regret, linucb_error=linucb_model.run()
	lints_regret, lints_error=lints_model.run()
	e_regret, e_error=e_model.run()

	lse_soft_regret, lse_soft_error, lse_soft_prob_matrix, lse_soft_s_matrix, lse_soft_g_s_matrix=lse_soft_model.run(user_feature, item_features, true_payoffs)
	online_regret, online_error, online_soft_matrix, online_s_matrix, online_beta_list, online_len_l_set, online_gradient_list=online_model.run()

	linucb_regret_matrix[l]=linucb_regret
	lints_regret_matrix[l]=lints_regret
	lse_soft_regret_matrix[l]=lse_soft_regret
	online_regret_matrix[l]=online_regret
	offline_prob_matrix[l]=lse_soft_prob_matrix[best_arm]
	online_prob_matrix[l]=online_soft_matrix[best_arm]
	online_beta_matrix[l]=online_beta_list
	e_regret_matrix[l]=e_regret


linucb_mean=np.mean(linucb_regret_matrix, axis=0)
linucb_std=linucb_regret_matrix.std(0)

lints_mean=np.mean(lints_regret_matrix, axis=0)
lints_std=lints_regret_matrix.std(0)

lse_soft_mean=np.mean(lse_soft_regret_matrix, axis=0)
lse_soft_std=lse_soft_regret_matrix.std(0)

online_mean=np.mean(online_regret_matrix, axis=0)
online_std=online_regret_matrix.std(0)


offline_prob_mean=np.mean(offline_prob_matrix, axis=0)
offline_prob_std=offline_prob_matrix.std(0)

online_prob_mean=np.mean(online_prob_matrix, axis=0)
online_prob_std=online_prob_matrix.std(0)

online_beta_mean=np.mean(online_beta_matrix, axis=0)
online_beta_std=online_beta_matrix.std(0)


e_mean=np.mean(e_regret_matrix, axis=0)
e_std=e_regret_matrix.std(0)

# np.save(path+'offline_prob_mean_d_%s'%(dimension), offline_prob_mean)
# np.save(path+'offline_prob_std_d_%s'%(dimension), offline_prob_std)
# np.save(path+'online_prob_mean_d_%s'%(dimension), online_prob_mean)
# np.save(path+'online_prob_std_d_%s'%(dimension), online_prob_std)

# np.save(path+'online_beta_mean_d_%s'%(dimension), online_beta_mean)
# np.save(path+'online_beta_std_d_%s'%(dimension), online_beta_std)

color_list=matplotlib.cm.get_cmap(name='Set1', lut=None).colors
x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', color=color_list[0], markevery=0.1, linewidth=2, markersize=5, label='LinUCB')
# plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lints_mean, '-', color=color_list[1], markevery=0.1, linewidth=2, markersize=5, label='LinTS')
# plt.fill_between(x, lints_mean-lints_std*0.95, lints_mean+lints_std*0.95, color='g', alpha=0.2)
plt.plot(x, e_mean, '-*', color=color_list[2], markevery=0.1, linewidth=2, markersize=5, label=r'$\epsilon$'+'-Greedy')
# plt.fill_between(x, e_mean-e_std*0.25, e_mean+e_std*0.25, color='c', alpha=0.2)
plt.plot(x, lse_soft_mean, '-|', color=color_list[3], markevery=0.1, linewidth=2, markersize=5, label='SoftUCB offline')
# plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='r', alpha=0.2)
plt.plot(x, online_mean, '-o', color=color_list[4], markevery=0.1, linewidth=2, markersize=5, label='SoftUCB online')
# plt.fill_between(x, online_mean-online_std*0.95, online_mean+online_std*0.95, color='orange', alpha=0.2)
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'simu_expucb_d_%s_item_%s'%(dimension, item_num)+'.png', dpi=300)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(linucb_error, color='b',linewidth=2, label='LinUCB')
plt.plot(lints_error,color='g', linewidth=2, label='LinTS')
plt.plot(lse_soft_error, color='r', linewidth=2, label='ExpUCB (offline)')
plt.plot(online_error, color='orange',linewidth=2, label='ExpUCB (online)')
plt.plot(e_error, color='c', linewidth=2, label='E-greedy')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.legend(loc=1, fontsize=12)
plt.tight_layout()
plt.savefig(path+'simu_error_shadow_soft_d_%s'%(dimension)+'.png', dpi=300)
plt.show()














