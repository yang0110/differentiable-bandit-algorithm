import numpy as np 
from scipy.special import softmax
from numpy.random import choice
from sklearn.preprocessing import Normalizer, MinMaxScaler


class Expucb_online():
	def __init__(self, dimension, iteration, item_num, user_feature, item_features, true_payoffs, alpha, sigma, step_size_beta,  weight1, beta):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=user_feature
		self.true_payoffs=true_payoffs
		self.item_features=item_features
		self.alpha=alpha
		self.sigma=sigma
		self.gamma=0
		self.weight1=weight1
		self.beta=beta
		self.a=step_size_beta
		self.step_size_beta=step_size_beta
		self.user_f=np.zeros(self.dimension)
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.s_list=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		self.low_bound_list=np.zeros(self.item_num)
		self.soft_max_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.est_y_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_grad=0
		self.lagrange_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.s_matrix=np.zeros((self.item_num, self.iteration))
		self.u_set=[]
		self.l_set=[]
		self.est_total_y=np.zeros(self.iteration)
		self.temp_1=0
		self.old_gradient=0
		self.temp_2=0
		self.temp_3=0


	def update_feature(self,x, y, index, time):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def select_arm(self, time):
		self.s_list=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		self.est_y_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		self.low_bound_list=np.zeros(self.item_num)
		for i in range(self.item_num):
			x=self.item_features[i]
			self.x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.est_y_list[i]=np.dot(self.user_f, x)
			self.low_bound_list[i]=np.dot(self.user_f, x)-self.beta*np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.est_y_matrix[time, i]=np.dot(self.user_f, x)

		for j in range(self.item_num):
			index=np.argmax(self.low_bound_list)
			self.x_norm_list2[j]=self.x_norm_list[j]+self.x_norm_list[index]
			self.s_list[j]=self.beta*(self.x_norm_list2[j])-np.abs(self.est_y_list[index]-self.est_y_list[j])
			self.s_matrix[j, time]=self.s_list[j]
			if self.s_list[j]>=0:
				self.u_set.extend([j])
			else:
				self.l_set.extend([j])

		soft_max=np.exp(self.gamma*self.s_list)/np.sum(np.exp(self.gamma*self.s_list))
		self.soft_max_matrix[time]=soft_max
		ind=choice(range(self.item_num), p=soft_max)
		x=self.item_features[ind]
		payoff=self.true_payoffs[ind]+np.random.normal(scale=self.sigma)
		regret=np.max(self.true_payoffs)-self.true_payoffs[ind]
		return ind, x, payoff, regret

	def find_lagrange_grad(self, time):
		for i in range(self.item_num):
			self.lagrange_grad_matrix[time, i]=self.weight1*self.x_norm_list[i]


	def find_log_grad_beta(self, time):
		a=self.gamma*np.dot(self.x_norm_list2, np.exp(self.gamma*self.s_list))
		b=np.sum(np.exp(self.gamma*self.s_list))
		for i in range(self.item_num):
			self.beta_log_grad_matrix[time, i]=self.gamma*(self.x_norm_list2[i])-a/b


	def update_gamma(self, time):
		cl=len(np.unique(self.l_set))
		delta=0.99
		if cl==0:
			cl=1
		else:
			pass
		a=np.log((delta*(cl))/(1-delta))
		max_s=np.max(self.s_list)
		min_s=np.min(self.s_list)
		self.gamma=a/max_s
		# print('cl, a,  gamma, max_s, min_s', np.round([cl, a, self.gamma]), np.round(max_s,decimals=2), np.round(min_s,decimals=2))

	def update_beta_grad(self, time, index, y):

		# self.temp_1+=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_list, self.soft_max_matrix[time], self.beta_log_grad_matrix[time], self.lagrange_grad_matrix[time])])
		self.temp_3+=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.beta_log_grad_matrix[time])])
		kk=np.sum([a*b*c for a,b,c in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.beta_log_grad_matrix[time])])
		self.temp_1=self.temp_3-kk
		self.temp_1+=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_list, self.soft_max_matrix[time], self.beta_log_grad_matrix[time], self.lagrange_grad_matrix[time])])

		# a=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.beta_log_grad_matrix[time], self.lagrange_grad_matrix[time])])
		#temp_2=np.sum(discound_list*a)

		a=np.sum([a*b*c for a,b,c in zip(self.est_y_list, self.soft_max_matrix[time], self.beta_log_grad_matrix[time])])

		discound_list=np.ones(self.iteration-time)
		for i in range(len(discound_list)):
			discound_list[i]=1**(i)

		self.temp_2=np.sum(discound_list*a)
		self.beta_grad=(self.temp_1+self.temp_2)/self.iteration

	def update_beta(self, index, time):
		self.beta+=self.step_size_beta*self.beta_grad


	def run(self):
		error_list=np.zeros(self.iteration)
		cum_regret=[0]
		beta_list=np.zeros(self.iteration)
		reward_list=np.zeros(self.iteration)
		gradient_list=np.zeros(self.iteration)
		l_set=np.zeros(self.iteration)
		temp1_list=np.zeros(self.iteration)
		temp2_list=np.zeros(self.iteration)
		gamma_list=np.zeros(self.iteration)
		for time in range(self.iteration):
			print('time/iteration, %s/%s beta=%s ~~~~~ ExpUCB-Online'%(time, self.iteration, np.round(self.beta, decimals=2)))
			ind, x, y, regret=self.select_arm(time)
			reward_list[time]=y
			self.update_feature(x, y, ind, time)
			self.est_total_y[time]=np.sum(reward_list[:time])+(self.iteration-time)*np.sum(self.est_y_matrix[time]*self.soft_max_matrix[time])
			self.update_gamma(time)
			self.find_log_grad_beta(time)
			self.find_lagrange_grad(time)
			self.update_beta_grad(time, ind, y)
			self.update_beta(ind, time)
			temp1_list[time]=self.temp_1
			temp2_list[time]=self.temp_2
			beta_list[time]=self.beta
			gradient_list[time]=self.beta_grad
			l_set[time]=len(np.unique(self.l_set))
			gamma_list[time]=self.gamma
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=np.linalg.norm(self.user_f-self.user_feature)

		return cum_regret[1:], error_list, self.soft_max_matrix.T, self.s_matrix, beta_list, temp1_list, temp2_list

		














