import bgd
verbose_test = True
"""This script verifies functionality of important functions used in Byzantine Gradient Descent"""
#=============================================================================================
def chk_dot_prod():
	a = [1, 2, 3]
	b = [4, 5, 6]
	print('Dot product works: ', bgd.dot_prod(a, b) == 4+10+18) if verbose_test else print('', end = '')
#=============================================================================================
def chk_calc_gradient():
	data = [[5, 7, 45], [4, 7, 47], [6, 7, 41]]
	theta = [3, 4]
	mac = bgd.Machine(data)
	print('Calculate gradient works: ', mac.calc_gradient(theta) == [-8, -28]) if verbose_test else print('', end = '')
#=============================================================================================
def chk_calc_mean(server):
	grad_li = [[5, 6], [2, 3], [4, 7], [8, 5], [9, 1]]
	print('Calculate mean works: ', server.calc_mean(grad_li) == [28/5, 22/5]) if verbose_test else print('', end = '')
#=============================================================================================
def chk_descent_step(server):
	with open ('config.txt', 'r') as config:
		lines = config.readlines()
		step_size = int(lines[3].split('=')[1].strip())
	prev_theta = [7, 6]
	gradient = [3, 4]
	server.theta_li.append(prev_theta)
	print('Descent step works: ', server.descent_step(gradient) == [7-step_size*3, 6-step_size*4]) if verbose_test else print('', end = '')
#=============================================================================================
def main():
	chk_dot_prod()
	chk_calc_gradient()
	server = bgd.Parameter_server()
	chk_calc_mean(server)
	chk_descent_step(server)

if __name__ == "__main__":
	main()
#=============================================================================================
