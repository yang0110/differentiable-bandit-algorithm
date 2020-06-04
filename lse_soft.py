import numpy as np 
from scipy.special import softmax
from numpy.random import choice
from sklearn.preprocessing import Normalizer, MinMaxScaler
# np.random.seed(2018)

class LSE_soft():
	def __init__(self, dimension, iteration, item_num, user_feature, alpha, sigma, step_size_beta, step_size_gamma, weight1, beta, gamma):
		self.dimension=dimension
		self.iteration=iteration
		self.item_num=item_num 
		self.user_feature=user_feature
		cov_matrix=self.dimension*np.identity(self.dimension)
		# +np.random.normal(size=(dimension, dimension))
		cov_inv=np.linalg.pinv(cov_matrix)
		self.random_item_features=np.random.multivariate_normal(mean=np.zeros(self.dimension), cov=cov_inv, size=1*self.item_num)
		self.random_item_features=Normalizer().fit_transform(self.random_item_features)
		self.alpha=alpha
		self.sigma=sigma
		self.weight1=weight1
		self.beta=beta
		self.gamma=gamma
		self.step_size_beta=step_size_beta
		self.step_size_gamma=step_size_gamma
		self.user_f=np.zeros(self.dimension)
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.s_list=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		self.low_bound_list=np.zeros(self.item_num)
		self.soft_max_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.gamma_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.est_y_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_grad=0
		self.gamma_grad=0
		self.temp_2=0
		self.lagrange_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.g_s_matrix=np.zeros((self.item_num, self.iteration))


	def initial(self):
		self.user_f=np.zeros(self.dimension)
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.s_list=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		self.low_bound_list=np.zeros(self.item_num)
		self.soft_max_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.gamma_log_grad_matrix=np.zeros((self.iteration, self.item_num))
		self.est_y_matrix=np.zeros((self.iteration, self.item_num))
		self.beta_grad=0
		self.gamma_grad=0
		self.gamma=0
		self.s_matrix=np.zeros((self.item_num, self.iteration))
		self.u_set=[]
		self.l_set=[]
		self.bound_matrix=np.zeros((self.item_num, self.iteration))

	def update_feature(self,x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def select_arm(self, time, old_payoff):
		self.s_list=np.zeros(self.item_num)
		temp=np.zeros(self.item_num)
		self.x_norm_list=np.zeros(self.item_num)
		self.x_norm_list2=np.zeros(self.item_num)
		est_y_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		self.low_bound_list=np.zeros(self.item_num)
		for i in range(self.item_num):
			x=self.item_features[i]
			self.x_norm_list[i]=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			est_y_list[i]=np.dot(self.user_f, x)
			self.low_bound_list[i]=np.dot(self.user_f, x)-self.beta*np.sqrt(np.dot(np.dot(x, cov_inv),x))
			self.est_y_matrix[time, i]=np.dot(self.user_f, x)

		for j in range(self.item_num):
			index=np.argmax(self.low_bound_list)
			self.x_norm_list2[j]=self.x_norm_list[j]+self.x_norm_list[index]
			self.s_list[j]=self.beta*(self.x_norm_list2[j])-np.abs(est_y_list[index]-est_y_list[j])
			self.s_matrix[j, time]=self.s_list[j]
			self.g_s_matrix[j,time]=self.gamma*self.s_list[j]
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


	def update_gamma(self):
		cl=len(np.unique(self.l_set))
		delta=0.6
		if cl==0:
			cl=1
		else:
			pass
		a=np.log((delta*cl)/(1-delta))
		max_s=np.max(self.s_list)
		self.gamma=a/max_s
		# print('cl', cl)
		# if cl==0:
		# 	self.gamma=1
		# self.gamma=1

	def update_beta_grad(self, time, index):
		temp_2=np.sum([a*b*c+d for a,b,c,d in zip(self.est_y_matrix[time], self.soft_max_matrix[time], self.beta_log_grad_matrix[time], self.lagrange_grad_matrix[time])])
		self.beta_grad=temp_2

	def update_beta(self, l, train_loops):
		# if (1/(l+1))>self.step_size_beta:
		# 	pass
		# else:
		# 	self.step_size_beta=1/(l+1)
		# if l>200:
		# 	self.step_size_beta=0.1
		# if l>train_loops/2:
		# 	self.step_size_beta=0.5
		self.beta+=self.step_size_beta*self.beta_grad


	def generate_data(self, item_num):
		random_items=choice(range(1*item_num),size=item_num, replace=False)
		self.item_features=self.random_item_features[random_items]
		self.true_payoffs=np.dot(self.item_features, self.user_feature)

	def train(self, train_loops, item_num, user_feature=None, item_features=None):
		self.cum_regret_loop=np.zeros(train_loops)
		self.beta_loop=np.zeros(train_loops)
		self.gamma_loop=np.zeros(train_loops)
		self.train_loops=train_loops
		self.beta_gradient_list=np.zeros(train_loops)
		for l in range(train_loops):
			if user_feature==None:
				self.generate_data(item_num)
			else:
				self.item_features=item_features
				self.user_feature=user_feature
			self.initial()
			error_list=np.zeros(self.iteration)
			cum_regret=[0]
			beta_list=np.zeros(self.iteration)
			gamma_list=np.zeros(self.iteration)
			old_payoff=0
			for time in range(self.iteration):
				print('Train-loop=%s, beta, gamma, time/iteration, %s/%s, %s, %s~~~~~ LSE-Soft'%(l, time, self.iteration, np.round(self.beta, decimals=2), np.round(self.gamma, decimals=2) ))
				ind, x, y, regret=self.select_arm(time, old_payoff)
				old_payoff=y
				self.update_feature(x, y)
				self.find_log_grad_beta(time)
				self.find_lagrange_grad(time)
				self.update_beta_grad(time, ind)
				cum_regret.extend([cum_regret[-1]+regret])
				error_list[time]=np.linalg.norm(self.user_f-self.user_feature)
				self.update_gamma()
			self.update_beta(l, train_loops)
			self.beta_gradient_list[l]=(1/train_loops)*self.beta_grad
			self.beta_loop[l]=self.beta
			self.gamma_loop[l]=self.gamma
			self.cum_regret_loop[l]=cum_regret[-1]
			
		return self.cum_regret_loop, self.beta_loop, self.soft_max_matrix.T, self.beta_gradient_list

	def run(self, user_feature, item_features, true_payoffs):
		self.user_feature=user_feature
		self.item_features=item_features
		self.true_payoffs=true_payoffs
		self.initial()
		error_list=np.zeros(self.iteration)
		cum_regret=[0]
		old_payoff=0
		for time in range(self.iteration):
			print('Test, time/iteration, %s/%s ~~~~~ LSE-Soft'%(time, self.iteration))
			ind, x, y, regret=self.select_arm(time, old_payoff)
			old_payoff=y
			self.update_feature(x, y)
			self.update_gamma()
			cum_regret.extend([cum_regret[-1]+regret])
			error_list[time]=np.linalg.norm(self.user_f-self.user_feature)

		return cum_regret[1:], error_list, self.soft_max_matrix.T, self.s_matrix, self.g_s_matrix

		














