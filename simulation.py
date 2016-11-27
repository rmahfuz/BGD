"""
Fault-tolerant distributed optimization: Admissibility Check
Author: Rehana Mahfuz
Date: 11/24/2016

Structure of data_p for each Agent: (CARDINALITY = 5 in this example)
           |data point 1|data point 2|data point 3|data point 4|data point 5|
_____________________________________________________________________________
x          |            |            |            |            |            |
y          |            |            |            |            |            |
labels     |            |            |            |            |            |
centers_x  |            |            |            |            |            |
centers_y  |            |            |            |            |            |
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
#from ctypes import *

CARDINALITY = 5
NUM_AGENTS = 15
NUM_LOCAL_ESTIMATES = 8
lambdaa = 1#chose this variable name because lambda seemed to be already taken

"""Function gradient
Functionality: Calculates gradient of a given value, by dividing by 2 (to be modified later), because I couldn't think enough about how to calculate gradient. But I realize that it has to be different for each agent, which is how each agetnt's local estimate will be different each time
Parameters: value of which you want gradient
Returns: calculated gradient
"""
def gradient(value):
    gradient = value/2
    return gradient

"""Class Agent
Instatiation expectations: pass 'data', [row_x, row_y, row_labels, center_x, center_y], each with CARDINALITY number of columns
"""
class Agent:
    myarray = np.zeros((NUM_LOCAL_ESTIMATES + 5,CARDINALITY)) #data points
    #rownames = ['x', 'y', 'label', 'ilocal_cost', 'xi[0]', 'xi[1]']
    #colnames = ['point1', 'point2', 'point3', 'point4', 'point5']
    data_p = pandas.DataFrame(myarray)
    #local_cost = 0;
    #Constructor:
    def __init__(self, data): #data is expected to be [row_x, row_y, row_labels, center_x, center_y], each with CARDINALITY number of columns
         self.data_p.iloc[:5,:CARDINALITY] = data
         
    def calc_LocalCost(self):
        for i in range(0,CARDINALITY):
            Agent.data_p.iloc[5,i] = math.sqrt(math.pow(((Agent.data_p.iloc[0,i])-(Agent.data_p.iloc[3,i])),2) + math.pow(((Agent.data_p.iloc[1,i]) - (Agent.data_p.iloc[4,i])),2))
        return [self.data_p.iloc[5,:]]
    #To be changed later:
    for i in range(0,CARDINALITY):
            data_p.iloc[6,i] = 0
    
    def calc_LocalEstimate(self,prev,time):#prev is a 1 x NUM_AGENTS array containing all agents' local estimates at the previous time. time is any nonzero positive integer
        #self.data_p.iloc[time+6,1] = sum(np.transpose(prev))/NUM_AGENTS - lambdaa*gradient(np.sum(prev)/NUM_AGENTS)
        return [self.data_p.iloc[time+6,1]]
        
    
"""Function calc_centers
Functionality: Calculates centers of a given data set, given the labels. Center is calculated as average of all points in that cluster.
Parameters: xy: 2 x CARDINALITY array. First row stores x values of data set. Second row stores y values.
lables: 1 x CARDINALITY array containing numerical labels for each data point.
Returns: 2 x CARDINALITY array of centers for each cluster. First row stores x values of the center. Second row stores y values of the center.
"""
def calc_centers(xy,labels):
    centers = np.zeros((2,CARDINALITY))
    for i in range(0,2):
        acc_x = 0
        acc_y = 0
        for j in range(0,CARDINALITY):
            if(labels[j] == i):
                acc_x += xy[0,j]
                acc_y += xy[1,j]
        for m in range(0,CARDINALITY):
            if(labels[m] == i):
                centers[0,m] = acc_x/CARDINALITY
                centers[1,m] = acc_y/CARDINALITY
    return centers


"""Function make_agent
Functionality: Makes an instance of the class Agent given a dataset with labels
Parameters: X1: 2 x CARDINALITY array. First row contains x, the first feature. Second row contains y, the second feature.
labels: 1 x CARDINALITY array containing numerical labels for each data point.
seed: seed for random initialization
Returns: the data for agent to be made
"""
def make_agent(X1,labels, seed):
    rn.seed(seed)
    data = X1[rn.sample(list(range(1,99)),CARDINALITY)]
    #print(data)
    data = np.transpose(np.array(data))
    rn.seed(seed)
    temp = labels[rn.sample(list(range(1,99)),CARDINALITY)]
    temp = np.transpose(np.array(temp))
    #print(temp)
    data = np.row_stack((data,temp))#added labels, so now 3 rows
    centers = calc_centers(data[0:2,:],data[2,:])
    data = np.row_stack((data,centers))#added centers, so now 5 rows
    #print(centers)
    #agent1 = Agent(data)
    return data
    
X1, labels = make_gaussian_quantiles(n_features=2, n_classes=3) #generating the gaussian distribution
agentList = []
for k in range(0,NUM_AGENTS):
    agentList.append(Agent(make_agent(X1, labels,k)))
    print(agentList[k].calc_LocalCost())

prev = np.zeros((1,NUM_AGENTS))
print(agentList[1].calc_LocalEstimate(prev,1))

    
"""#generating gaussian distribution:
X1, labels = make_gaussian_quantiles(n_features=2, n_classes=3)
rn.seed(10)
data = X1[rn.sample(list(range(1,99)),CARDINALITY)]
#print(data)
data = np.transpose(np.array(data))
rn.seed(10)
temp = labels[rn.sample(list(range(1,99)),CARDINALITY)]
temp = np.transpose(np.array(temp))
#print(temp)
data = np.row_stack((data,temp))#added labels, so now 3 rows
centers = calc_centers(data[0:2,:],data[2,:])
data = np.row_stack((data,centers))#added centers, so now 5 rows
#print(centers)
agent1 = Agent(data)
print("Local costs:\n:")
print(agent1.calc_LocalCost(centers))"""


                















        
    
