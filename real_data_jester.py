import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("white")
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
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
input_path='../data/'
path='../results/real_data/'
np.random.seed(2018)


jester=np.load(input_path+'jester.npy')

total_item_num=100
item_features=jester[:1000,:36].copy()
# item_features=Normalizer().fit_transform(item_features)
rewards=jester[:1000,37].copy()
mms=MinMaxScaler()
rewards=mms.fit_transform(rewards.reshape(-1,1))
# plt.hist(rewards)
# plt.show()
dim=20
pca=PCA(n_components=dim)
item_features=pca.fit_transform(item_features)
# item_features=Normalizer().fit_transform(item_features)

cov=0.1*np.identity(dim)
bias=np.zeros(dim)
for i in range(1000):
	x=item_features[i]
	cov+=np.outer(x,x)
	bias+=rewards[i]*x

user_feature=np.dot(np.linalg.pinv(cov), bias)
user_feature=user_feature/np.linalg.norm(user_feature)
j_item=item_features.copy()



user_num=1
item_num=30
dimension=dim
phase_num=11
iteration=2**phase_num
sigma=0.1# noise
delta=0.1# high probability
alpha=1 # regularizer
step_size_beta=0.1
step_size_gamma=0.01
weight1=0.003
loop=5

beta=0.4
gamma=0
train_loops=300

beta_online=2.2

v=0.3


linucb_regret_matrix=np.zeros((loop, iteration))
lints_regret_matrix=np.zeros((loop, iteration))
giro_regret_matrix=np.zeros((loop, iteration))
lse_soft_regret_matrix=np.zeros((loop, iteration))
online_regret_matrix=np.zeros((loop, iteration))
offline_prob_matrix=np.zeros((loop, iteration))
online_prob_matrix=np.zeros((loop, iteration))
online_beta_matrix=np.zeros((loop, iteration))
e_regret_matrix=np.zeros((loop, iteration))


# train model
lse_soft_model=LSE_soft(dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma)

# lse_soft_regret_list_train, lse_soft_beta_list_train=lse_soft_model.train(train_loops, item_num)
random_item=np.random.choice(range(100), size=item_num)
item_features=j_item[random_item]
true_payoffs=rewards[random_item]
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



color_list=matplotlib.cm.get_cmap(name='Set1', lut=None).colors
x=range(iteration)
plt.figure(figsize=(5,5))
plt.plot(x, linucb_mean, '-.', color=color_list[0], markevery=0.1, linewidth=2, markersize=5, label='LinUCB')
# plt.fill_between(x, linucb_mean-linucb_std*0.95, linucb_mean+linucb_std*0.95, color='b', alpha=0.2)
plt.plot(x, lints_mean, '-', color=color_list[1], markevery=0.1, linewidth=2, markersize=5, label='LinTS')
# plt.fill_between(x, lints_mean-lints_std*0.95, lints_mean+lints_std*0.95, color='g', alpha=0.2)
plt.plot(x, e_mean, '-*', color=color_list[2], markevery=0.1, linewidth=2, markersize=5, label=r'$\epsilon$'+'-Greedy')
# plt.fill_between(x, e_mean-e_std*0.25, e_mean+e_std*0.25, color='c', alpha=0.2)
# plt.plot(x, exp3_mean, '-|', color='gray', markevery=0.1, linewidth=2, markersize=8, label='Exp3')
# plt.fill_between(x, exp3_mean-exp3_std*0.25, exp3_mean+exp3_std*0.25, color='gray', alpha=0.2)
plt.plot(x, lse_soft_mean, '-|', color=color_list[3], markevery=0.1, linewidth=2, markersize=5, label='SoftUCB offline')
# plt.fill_between(x, lse_soft_mean-lse_soft_std*0.95, lse_soft_mean+lse_soft_std*0.95, color='r', alpha=0.2)
plt.plot(x, online_mean, '-o', color=color_list[4], markevery=0.1, linewidth=2, markersize=5, label='SoftUCB online')
# plt.fill_between(x, online_mean-online_std*0.95, online_mean+online_std*0.95, color='orange', alpha=0.2)
# plt.ylim([0,200])
plt.legend(loc=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Cumulative Regret', fontsize=14)
plt.tight_layout()
plt.savefig(path+'jester_expucb_d_%s_item_%s'%(dimension, item_num)+'.png', dpi=300)
plt.show()

