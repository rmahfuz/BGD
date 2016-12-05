"""
Fault-tolerant distributed optimization: Admissibility Check
Author: Rehana Mahfuz
Date: 11/24/2016

Structure of data_p for each Agent: (CARDINALITY = 5 in this example)
           |data point 1|data point 2|data point 3|data point 4|data point 5|
_____________________________________________________________________________
x          |            |            |            |            |            |
y          |            |            |            |            |            |
#labels     |            |            |            |            |            |
#centers_x  |            |            |            |            |            |
#centers_y  |            |            |            |            |            |
ilocal_cost|            |            |            |            |            |  (individual local cost) (only the first column stores data. The rest of it is unnecessary. I just realized. Sorry for allocating so much space. But if I change right now, everything will get messed up
xi[0]      |            |            |            |            |            |  (local estimate at t = 0)
xi[1]      |            |            |            |            |            |  (local estimate at t = 1)
xi[2]      |            |            |            |            |            |  (local estimate at t = 2)
xi[3]      |            |            |            |            |            |  (local estimate at t = 3)
xi[4]      |            |            |            |            |            |  (local estimate at t = 4)
xi[5]      |            |            |            |            |            |  (local estimate at t = 5)
xi[6]      |            |            |            |            |            |  (local estimate at t = 6)
xi[7]      |            |            |            |            |            |  (local estimate at t = 7)
_____________________________________________________________________________
"""
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import pandas
import math
import random as rn
import matplotlib.pyplot as plt
import copy as cp
#from ctypes import *

CARDINALITY = 10
NUM_AGENTS = 10
NUM_LOCAL_ESTIMATES = 8
BETA = 1
NUM_SIM = 30 #numbr of simulations to run (from t = 0 to t = NUM_SIM)
lambdaa = 1#chose this variable name because lambda seemed to be already taken

"""Class Agent
Instatiation expectations: pass 'data', [row_x, row_y, row_labels, center_x, center_y], each with CARDINALITY number of columns
"""
class Agent:
    #data_p = np.zeros((2,CARDINALITY)) #data points
    #gradient = 0
    #init_est = 0
    #Constructor:
    def __init__(self, data): #data is expected to be [row_x, row_y, row_labels, center_x, center_y], each with CARDINALITY number of columns
         self.data_p = np.zeros((2,CARDINALITY))
         self.data_p[:,:] = data
         #print(data)
         self.init_est = (sum(self.data_p[1,:])/sum(self.data_p[0,:]))
         
         #print(self.init_est)
         
    def calc_gradient(self,b):
        sum = 0
        #print(self.data_p)
        #print('\n')

        for i in range(0,CARDINALITY):
            sum += self.data_p[1,i] - b*self.data_p[0,i]
            #print(sum)
        #print('\n')
        self.gradient = sum /CARDINALITY
        return (sum/CARDINALITY)
    def disp(self):
        print(self.data_p)
        #print(self.gradient)
    

"""Function make_agent
Functionality: Makes an instance of the class Agent given a dataset with labels
Parameters: seed: seed for random initialization
Returns: the data for agent to be made
"""
def make_agent(seed):
    neg =0
    pos =0
    rn.seed(seed)
    x = rn.sample(list(range(1,100)),CARDINALITY)
    y = [0]*CARDINALITY
    for i in range(0,CARDINALITY):
        x[i]/=100
    for j in range(0,CARDINALITY):
        rn.seed(seed+j)
        w = rn.sample(list(range(0,100)),1)#gaussian noise between 0 and 1
        power = rn.randrange(-1,2,1)
        #print(pow(-1,power))
        if pow(-1,power) == 1:
            pos += 1
        else:
            neg += 1
        #print(' | ')
        w[0] /= 100*pow(-1,power)
        y[j] = BETA*x[j] + w[0]
    data = np.zeros((2,CARDINALITY)) #data points
    data[0,:] = x
    data[1,:] = y
    print(pos, neg)
    #print("neg = \n",neg)
    #print(data)
    #print('\n')
    return data

#make_agent(10)
#agentList = []
#agentList.append([Agent(make_agent(0))])
agentList = [cp.copy(x) for x in [0]*NUM_AGENTS]
agentList[0] = Agent(make_agent(0))

for k in range(1,NUM_AGENTS):
    #agentList[k-1].disp()
    #agentList[k].calc_gradient(3)
    #new_agent = cp.deepcopy(agentList[k-1])
    new_agent = Agent(make_agent(2*k))
    agentList[k] = new_agent
    #new_agent.disp()
    #agentList.append(new_agent)
    #agentList[k].disp()
    #for i in range(0,k+1):
    #   print(agentList[i].data_p)
    #print('\n')
    
"""for b in range(0,NUM_AGENTS):
    print(agentList[b].data_p)
    #print(agentList[b].calc_gradient(3))
    print('\n')"""
    

local_est = np.zeros((NUM_SIM,NUM_AGENTS))
for i in range(0,NUM_AGENTS):
    local_est[0,i] = agentList[i].init_est
    #print(i)
    #print(agentList[i].data_p)
    #print('\n')
    
terminate = 0;    
for t in range(1,NUM_SIM):
    avg_b = sum(local_est[t-1,:])/NUM_AGENTS
    for j in range(0,NUM_AGENTS):
        grad = agentList[j].calc_gradient(avg_b)
        #print(grad)
        local_est[t,j] = avg_b - (1/t)*grad
        if (abs(local_est[t,j] - local_est[t-1,j])) < .000001:
            terminate = 1
            break
    if terminate == 1:
        break
        #print(local_est[t,:])
    #if (local_est[t,j] - local_est[t-1,j]) < .00001:
     #   break

for c in range(0,NUM_SIM):
    print(c)
    print(local_est[c,:])
    print('\n')

plt.plot(range(0,NUM_SIM),local_est)
plt.show()
    


    











        
    
