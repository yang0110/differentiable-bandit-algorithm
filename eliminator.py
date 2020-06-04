import numpy as np 

class ELI():
	def __init__(self, dimension, phase_num, item_num, user_feature, item_feature, true_payoffs, alpha, delta, sigma):
		self.dimension=dimension
		self.phase_num=phase_num
		self.iteration=2**phase_num
		self.item_num=item_num
		self.user_feature=user_feature
		self.item_feature=item_feature
		self.true_payoffs=true_payoffs
		self.alpha=alpha
		self.delta=delta
		self.sigma=sigma
		self.beta=np.sqrt(2*np.log(1/self.delta))+np.sqrt(self.alpha)
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.item_set=list(range(self.item_num))
		self.low_ucb_list=np.zeros(self.item_num)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.item_index=np.zeros(self.iteration)
		self.low_matrix=np.zeros((self.item_num, self.iteration))
		self.upper_matrix=np.zeros((self.item_num, self.iteration))
		self.payoff_error_matrix=np.zeros((self.item_num, self.iteration))
		self.worst_payoff_error=np.zeros(self.iteration)
		self.noise_norm=np.zeros(self.iteration)
		self.noise_bias=np.zeros(self.dimension)
		self.bound=np.zeros(self.iteration)	

	def update_error(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		bound_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv), x))
			bound_list[i]=self.beta*x_norm
		self.bound[time]=np.max(bound_list)

	def select_arm(self, time):
		x_norm_list=np.zeros(self.item_num)
		cov_inv=np.linalg.pinv(self.cov)
		for i in self.item_set:
			x=self.item_feature[i]
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv),x))
			x_norm_list[i]=x_norm

		max_index=np.argmax(x_norm_list)
		self.item_index[time]=max_index
		noise=np.random.normal(scale=self.sigma)
		payoff=self.true_payoffs[max_index]+noise
		regret=np.max(self.true_payoffs)-payoff+noise
		x=self.item_feature[max_index]
		x_best=self.item_feature[np.argmax(self.true_payoffs)]
		self.noise_bias+=x*noise 
		self.noise_norm[time]=np.abs(np.dot(x_best, np.dot(cov_inv, self.noise_bias)))
		return x, payoff, regret 

	def update_feature(self,x,y):
		self.cov+=np.outer(x,x)
		self.bias+=x*y
		self.user_f=np.dot(np.linalg.pinv(self.cov), self.bias)

	def update_bounds(self, time):
		cov_inv=np.linalg.pinv(self.cov)
		self.upper_ucb_list=np.zeros(self.item_num)
		self.low_ucb_list=np.zeros(self.item_num)
		for i in self.item_set:
			x=self.item_feature[i]
			est_y=np.dot(self.user_f, x)
			x_norm=np.sqrt(np.dot(np.dot(x, cov_inv), x))
			self.upper_ucb_list[i]=est_y+self.beta*x_norm 
			self.low_ucb_list[i]=est_y-self.beta*x_norm 
			self.payoff_error_matrix[i,time]=np.abs(self.true_payoffs[i]-est_y)
			self.upper_matrix[i, time]=est_y+self.beta*x_norm 
			self.low_matrix[i, time]=est_y-self.beta*x_norm 

	def reset(self):
		self.cov=self.alpha*np.identity(self.dimension)
		self.bias=np.zeros(self.dimension)
		self.user_f=np.zeros(self.dimension)
		self.noise_bias=np.zeros(self.dimension)

	def eliminate_arm(self):
		a=self.item_set.copy()
		for i in a:
			if np.max(self.low_ucb_list)>self.upper_ucb_list[i]:
				self.item_set.remove(i)
			else:
				pass 

	def run(self):
		cum_regret=[0]
		error=np.zeros(self.iteration)
		error[0]=1
		for l in range(self.phase_num):
			start_time=2**l 
			end_time=2**(l+1)
			for time in range(start_time, end_time):
				print('time/iteration=%s/%s, item_num=%s ~~~~ Eliminator'%(time, self.iteration, len(self.item_set)))
				x,y, regret=self.select_arm(time)
				self.update_feature(x,y)
				self.update_error(time)
				self.update_bounds(time)
				cum_regret.extend([cum_regret[-1]+regret])
				error[time]=np.linalg.norm(self.user_f-self.user_feature)
				self.worst_payoff_error[time]=np.max(self.payoff_error_matrix[:,time])
			self.eliminate_arm()
			self.reset()

		return cum_regret, error, self.item_index, self.upper_matrix, self.low_matrix, self.payoff_error_matrix, self.worst_payoff_error, self.noise_norm, self.bound



























