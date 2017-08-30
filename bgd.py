"""Byzantine gradient descent"""

import matplotlib.pyplot as plt
import numpy as np
from geometric_median import geometric_median

#==================================================================================================
"""Reading from config file"""
with open ('config.txt', 'r') as config:
	lines = config.readlines()
	num_samples = int(lines[0].split('=')[1].strip())
	num_machines = int(lines[1].split('=')[1].strip())
	num_batches = int(lines[2].split('=')[1].strip())
	step_size = int(lines[3].split('=')[1].strip())
	dimension = int(lines[4].split('=')[1].strip())
	theta_star_raw = lines[5].split('=')[1].strip()
	theta_star = theta_star_raw[1:len(theta_star_raw)-1].split(',')
	for i in range(len(theta_star)):
		theta_star[i] = int(theta_star[i])
	num_byz = int(lines[6].split('=')[1].strip()) #number of byzantine faults to be tolerated
	num_iter = int(lines[7].split('=')[1].strip())  #number of iterations
	verbose = bool(int(lines[12].split('=')[1]))
	if verbose:
		print("Configuration:")
		print("")
		for line in lines:
			print(line, end = "")

#==================================================================================================
def dot_prod(vec1, vec2):
	"""Calculates inner product
	   Accepts 2 vectors of equal length
	   Returns a scalar value"""
	if len(vec1) != len(vec2):
		raise ValueError('In calculating inner product, both arguments must be of the same length')
	sum_val = 0
	for i in range(len(vec1)):
		sum_val += vec1[i] * vec2[i]
	return sum_val
#==================================================================================================		
class Machine:
	def __init__(self, data):
		"""Initializes the machine with the data
		   Accepts data, a list of length 'num_samples/num_machines', containing data samples, each of length 'dimension+1'"""
		if verbose:
			print("\nMachine data:\n")
			for i in range(len(data)):
				print("Data sample %d: " % (i+1), end = '')
				#print(data[i])
				for j in range(len(data[i])):
					print("%10f|" % round(data[i][j], 3), end = '')
				print('')
		self.data = data
	def calc_gradient(self, theta):
		"""Calculates gradient of all local data samples, given theta
		   Accepts theta, a list of length 'dimension' 
		   Returns sum over all local data samples[(y-<w, theta>) * (-w)], list of length 'dimension'"""
		gradient = [0] * dimension #list to store the gradient of length dimension
		for val in self.data:
			w = val[:len(val) - 1] #all elements of the data sample except the last
			y = val[len(val) - 1] #last element of data sample
			scalar_val = y - dot_prod(w, theta)
			for i in range(dimension):
				gradient[i] += (scalar_val * (-w[i]))
		return gradient
#==================================================================================================		
class Parameter_server:
	#--------------------------------------------------------------------------
	def __init__(self):
		"""Initializes all machines"""
		self.theta_li = [] #list that stores each theta, grows by one each iteration
		"""
		with open ('config.txt', 'r') as config:
			lines = config.readlines()
			distro = lines[8].split('=')[1].strip()
			mean = int(lines[9].split('=')[1].strip())
			stddev = int(lines[10].split('=')[1].strip())
		"""
		##Generating the data...............................................................
		"""
		mean = [0, 0]
		stddev = [[1, 0], [0, 100]]
		"""
		with open('config.txt', 'r') as config:
			lines = config.readlines()
			mean_raw = lines[9].split('=')[1].strip()
			stddev_raw = lines[10].split('=')[1].strip()
		mean = mean_raw[1:len(mean_raw)-1].split(',')
		for i in range(len(mean)):
			mean[i] = int(mean[i])

		stddev = stddev_raw[1:len(stddev_raw)-1].split('(')
		stddev.remove('')
		for i in range(len(stddev)):
			stddev[i] = stddev[i].strip()
			if i != len(stddev) - 1:
				stddev[i] = stddev[i][0:len(stddev[i])-2].split(',')
			else:
				stddev[i] = stddev[i][0:len(stddev[i])-1].split(',')
			for j in range(len(stddev[i])):
				stddev[i][j] = int(stddev[i][j])
		data = np.random.multivariate_normal(mean, stddev, num_samples) #normal data, generated w
		data = data.tolist()
		for i in range(num_samples): #appending the 'y' to each data sample
			z = np.random.normal(0, 1) #noise
			data[i].append(dot_prod(data[i], theta_star) + z) #y = <w,theta*> + z, where X = (w, y)
		##Initializing the machines--........................................................
		samples_per_machine = int(num_samples / num_machines) #number of data samples per machine
		self.machines = [] #list containing worker machines
		for i in range(num_machines):
			#print('samples_per_machine = ', samples_per_machine)
			new_data = data[samples_per_machine*i : samples_per_machine*(i+1)]
			print('______________' + '\n' + 'Machine  '+str(i+1)) if verbose else print('', end = '')
			new_machine = Machine(new_data) #initializing a machine with the data
			self.machines.append(new_machine) 
		print('_________________________________________________________________________________________________________________________________') if verbose else print('', end = '')
	#--------------------------------------------------------------------------
	def broadcast(self, theta):
		"""Broadcasts theta
		   Accepts theta, a list of length 'dimension'
		   Returns a list of length 'num_machines' containing gradients from each machine, each gradient being of length 'dimension'"""
		grad_li = []
		for mac in self.machines:
			grad_li.append(mac.calc_gradient(theta))
		#print('gradient list =',grad_li) if verbose else print('', end = '')	
		if verbose:
			print('gradient list =',grad_li)
		else:
			print('', end = '')	
		return grad_li
	#--------------------------------------------------------------------------
	def calc_means(self, grad_li): 
		"""Calculates a list of means of gradients per batch, given the gradient list, for Byzantine gradient descent
		   Accepts gradient_list, a list of length 'num_machines' containing gradients, each gradient of length 'dimension'
		   Returns a list of length 'num_batches', containing means of gradients, each mean being a list of length 'dimension'"""
		mean_li = []
		batch_len = int(num_machines/num_batches) #number of machines per batch
		sum_val = [0] * dimension
		for cnt in range(num_machines):
			#----------------------------------------
			#adding values to the sum:
			for j in range(dimension):
				sum_val[j] += grad_li[cnt][j]
			#----------------------------------------
			cnt += 1 #incrementing count
			#----------------------------------------
			#appending average to mean list:
			if (cnt % batch_len) == 0:
				avg_val = [0] * dimension
				for j in range(dimension):
					avg_val[j] = sum_val[j] / batch_len
				mean_li.append(avg_val)
				sum_val = [0] * dimension
			#----------------------------------------
		return mean_li
	#--------------------------------------------------------------------------
	def calc_geo_median(self, li):
		"""Calculates geometric median of a given list"""
		#geo_median = li[int(len(li)/2)]
		geo_median = geometric_median(li)
		return geo_median
	#--------------------------------------------------------------------------
	def calc_mean(self, grad_li):
		"""Calculates average of the gradient list for standard gradient descent
		   Accepts a list of length num_machines containing gradients, each gradient being a list of length 'dimension'
		   Returns the average gradient, a list of length 'dimension'"""
		mean = [0] * dimension
		sum_val = [0] * dimension
		for i in range(len(grad_li)):
			for j in range(dimension):
				sum_val[j] += grad_li[i][j]
		for i in range(dimension):
			mean[i] = sum_val[i] / len(grad_li)
		return mean
	#--------------------------------------------------------------------------
	def descent_step(self, gradient):
		"""Performs the descent step given the gradient
		   Accepts gradient, a list of length 'dimension'
		   Returns new_theta, a list of length 'dimension'"""
		prod = [step_size * x for x in gradient] #multiplying the gradient (of dimension d) with scalar step_size (gradient*step_size)
		new_theta = []
		for i in range(len(prod)):
			new_theta.append(self.theta_li[-1][i] - prod[i]) #previous_theta - product
		print('new theta =', new_theta) if verbose else print('', end = '')
		return new_theta
	#--------------------------------------------------------------------------
	def gradient_descent(self, first_theta):
		"""Performs num_iter rounds of gradient descent, appends each new theta to theta_li
		   Accepts the first theta, a list of length 'dimension'"""
		self.theta_li.append(first_theta)
		for i in range(num_iter):
			print('Iteration %d:' %(i+1)) if verbose else print('', end = '')
			grad_li = self.broadcast(self.theta_li[i]) #same as self.theta_li[-1]
			with open ('config.txt', 'r')as config:
				lines = config.readlines()
				algorithm = str(lines[11].split('=')[1]).strip()
				#print 'algorithm = ', algorithm
			if algorithm == 'Byzantine':
				mean_li =  self.calc_means(grad_li)
				geo_median = self.calc_geo_median(mean_li)
				new_theta = self.descent_step(geo_median)
			if algorithm == 'Standard':
				mean = self.calc_mean(grad_li)
				new_theta = self.descent_step(mean)

			self.theta_li.append(new_theta)
			print('__________________________________________________________________________________') if verbose else print('', end = '')
	#--------------------------------------------------------------------------
	def plot(self):
		"""Plots theta against number of parameters"""
		#plot theta_li vs range(len(theta_li))
		print(self.theta_li)
		plt.plot(self.theta_li)
		plt.xlabel('Number of iterations')
		plt.ylabel('Learned value')
		plt.title('Learned value over iterations')
		plt.show()
	#--------------------------------------------------------------------------
#==================================================================================================
def init():
	server = Parameter_server()
	return server
#==================================================================================================
def main():
	server = init()
	server.gradient_descent([9, 10])
	print('\nTheta list\n')
	for theta in server.theta_li:
		print(theta)
	server.plot()
#==================================================================================================
if __name__ == "__main__":
	main()
#==================================================================================================
