#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:08:03 2020

@author: joelmcfarlane

Networks Project - Functions

"""
#%%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
#import seaborn as sns
from logbin230119 import *
from scipy.optimize import *
import sklearn.metrics as skm
from scipy.stats import chisquare
import scipy as sp
from tqdm import *
#sns.set()

#plt.grid()

col_list = ['b','g','r','c','m','orange','y','gray','darkorange','limegreen'\
            ,'aquamarine','lightsteelblue','teal','lightsalmon','olive',\
            'pink','dodgerblue','mediumslateblue','firebrick']


def which_node(edge_list,degree_list):
    """
    Function that picks a node to connect to with a probability proportional to
    the degree of the node.
    """
    sum_degree = np.sum(np.array(degree_list))
    probs = np.divide(degree_list,sum_degree)
    node_no = np.random.choice(edge_list,p = list(probs))
    return node_no

def frequency(G):
    " Calculate the probability of a degree k."
    
    freq = collections.Counter([val for (node, val) in G.G.degree()])
    df = pd.DataFrame.from_dict(freq, orient='index').reset_index()
    df.columns = ['Degree', 'Frequency']
    df = df.sort_values(by = ['Degree'])
    df['Probability'] = df['Frequency'] / \
    len([val for (node, val) in G.G.degree()])
    return df

def frequency_B(deg_list):
    freq = collections.Counter(list(deg_list))
    df = pd.DataFrame.from_dict(freq, orient='index').reset_index()
    df.columns = ['Degree', 'Frequency']
    df = df.sort_values(by = ['Degree'])
    df['Probability'] = df['Frequency'] / len(deg_list)
    return df

def linear(x,m,c):
    "Function for linear regression"
    y = m*x+c
    return y

class Graph():
    """ 
    Graph and functions for stage 1, pure preferential attachment. Note that 
    multiple edges are allowed in order to avoid minumum degree issues.
    """
    def __init__(self,m):
        
        self.time = m +1 
        self.G = nx.complete_graph(m+1,nx.MultiGraph())
        self.node_list = list(np.arange(0,m))
        # 1st node has to attach to only other node, so time starts at 1 and 
        # both nodes originally have one degree.
        self.degree_list = list(np.ones(m+1) * m)
        self.pref = list(np.array([(np.ones(m+1) * i) for i in range(m+1)]).flatten())
        self.m = m
    
    def drive(self):
        "Add a node and connect it to an existing node."
        self.time += 1
        other_side = np.random.choice(self.pref)
        self.G.add_node(self.time)
        not_this = []
        add_these = []
        for i in range(self.m):
            
            other_side = np.random.choice([x for x in self.pref if x not in not_this])
            add_these.append(other_side)
#            self.edge_list.append(self.time)
            self.G.add_edge(self.time,other_side)
            not_this.append(other_side)
#            self.degree_list[other_side] += 1
            
#        self.degree_list.append(m)
        for i in range(self.m):
            self.pref.append(add_these[i])
            self.pref.append(self.time)
            
    def draw(self):
        "Draw the Network and label it."
        pos = nx.spring_layout(self.G)
        nx.draw(self.G,with_labels = True,pos = pos)

    def add_N(self,N):
        "Add N nodes."
        for i in trange(N):
            self.drive()
            
            
class Graph_PA():
    """ Graph and functions for stage 1, pure preferential attachment."""
    def __init__(self):
        
        self.time = 1
        self.G = nx.Graph()
        self.edge_list = [0,1]
        self.G.add_node(0)
        self.G.add_node(1)
        # 1st node has to attach to only other node, so time starts at 1 and 
        # both nodes originally have one degree.
        self.degree_list = [1,1]
        self.G.add_edge(0,1)
        self.pref = [0,1]
    
    def drive(self,m):
        "Add a node and connect it to an existing node."
        self.time += 1
        other_side = np.random.choice(self.pref)
        self.G.add_node(self.time)
        for i in range(m):

            other_side = np.random.choice(self.pref)
            self.edge_list.append(self.time)
            self.G.add_edge(self.time,other_side)
#            print('boo')
            self.degree_list[other_side] += 1
            self.pref.append(other_side)
        self.degree_list.append(m)
        for i in range(m):
            self.pref.append(self.time)
            
    def draw(self):
        "Draw the Network and label it."
        pos = nx.spring_layout(self.G)
        nx.draw(self.G,with_labels = True,pos = pos)

    def add_N(self,N,m):
        "Add N nodes."
        for i in trange(N):
            self.drive(m)
            
class Graph_PA_fast():
    """
    Graph and functions without networkx package. Note this is a multigraph
    """
    def __init__(self,m):
        self.time = 1
        self.m = m
        self.node_list = list(np.arange(0,m+1,1))
        self.degree_list = list(np.ones(m+1)*m)
        self.pref = list(np.array([(np.ones(m+1) * i) for i in range(m+1)]).flatten())
    
    def drive(self):
        self.time += 1
        self.node_list.append(self.time)
        add_these = []
        self.degree_list.append(0)
        for i in range(self.m):
            other_side = int(np.random.choice(self.pref)) #[x for x in self.pref if x not in add_these]
            
            add_these.append(other_side)
            self.degree_list[-1] += 1
            self.degree_list[other_side] += 1
        self.pref = self.pref + add_these
        self.pref = self.pref + list(np.ones(self.m)*self.time)
        
    def add_N(self,N):
        for i in trange(N):
            self.drive()
        
                        
            
        
        
            
    
def prob_inf(k,m):
    "A model of the probability dist in the long time limit"
    p = (2*m*(m+1)) / (k*(k+1)*(k+2))
    return p

def power(k,A,s):
    return A*k**s      
            

class Graph_RA():
    """ Graph and functions for stage 2, pure random attachment."""
    def __init__(self,m):
        
        self.time = m
        self.G = nx.complete_graph(m+1,nx.Graph())
        self.node_list = list(np.arange(0,m))
        # 1st node has to attach to only other node, so time starts at 1 and 
        # both nodes originally have one degree.
        self.degree_list = list(np.ones(m+1) * m)
        self.m = m
#        self.pref = [0,1]
    
    def drive(self):
        "Add a node and connect it to an existing node."
        self.time += 1
#        prev_edge_list = self.edge_list
#        other_side = which_node(prev_edge_list,self.degree_list)
        other_side = np.random.choice(self.node_list)
        self.G.add_node(self.time)
        not_this = []
        for i in range(self.m):
#            other_side = which_node(prev_edge_list,self.degree_list)
            other_side = np.random.choice([x for x in self.node_list\
                                           if x not in not_this])
            self.G.add_edge(self.time,other_side)
            self.degree_list[other_side] += 1
            not_this.append(other_side)
        self.degree_list.append(self.m)
        self.node_list.append(self.time)
#            self.pref.append(other_side)
#        for i in range(m):
#            self.pref.append(self.time)
            
    def draw(self):
        "Draw the Network and label it."
        pos = nx.spring_layout(self.G)
        nx.draw(self.G,with_labels = True,pos = pos)

    def add_N(self,N):
        "Add N nodes."
        for i in trange(N):
            self.drive()

class Graph_RA_original():
    """ Graph and functions for stage 2, pure random attachment."""
    def __init__(self):
        
        self.time = 1
        self.G = nx.Graph()
        self.edge_list = [0,1]
        self.G.add_node(0)
        self.G.add_node(1)
        # 1st node has to attach to only other node, so time starts at 1 and 
        # both nodes originally have one degree.
        self.degree_list = [1,1]
        self.G.add_edge(0,1)
#        self.pref = [0,1]
    
    def drive(self,m):
        "Add a node and connect it to an existing node."
        self.time += 1
#        prev_edge_list = self.edge_list
#        other_side = which_node(prev_edge_list,self.degree_list)
        other_side = np.random.choice(self.edge_list)
        self.G.add_node(self.time)
        for i in range(m):
#            other_side = which_node(prev_edge_list,self.degree_list)
            other_side = np.random.choice(self.edge_list)
            self.G.add_edge(self.time,other_side)
            self.degree_list[other_side] += 1
        self.degree_list.append(m)
        self.edge_list.append(self.time)
#            self.pref.append(other_side)
#        for i in range(m):
#            self.pref.append(self.time)
            
    def draw(self):
        "Draw the Network and label it."
        pos = nx.spring_layout(self.G)
        nx.draw(self.G,with_labels = True,pos = pos)

    def add_N(self,N,m):
        "Add N nodes."
        for i in trange(N):
            self.drive(m)

class Graph_RWPA():
    """
    Graph and functions for stage 3, Random Walks and Preferential Attachement
    """
    def __init__(self,q = 0.5):
        
        self.time = 1
        self.G = nx.MultiGraph()
        self.node_list = [0,1]
        self.G.add_node(0)
        self.G.add_node(1)
        # 1st node has to attach to only other node, so time starts at 1 and 
        # both nodes originally have one degree.
        self.G.add_edge(0,1)
        self.q = q
#        self.pref = [0,1]
    
    def drive(self,m):
        "Add a node and connect it to an existing node."
        self.time += 1
#        prev_edge_list = self.edge_list
#        other_side = which_node(prev_edge_list,self.degree_list)
#        other_side = np.random.choice(self.node_list)
        self.G.add_node(self.time)
        
#        n0 = 1
        
#        self.G.add_edge(self.time,v0)
        
        for j in range(m):
                                                  
            v0 = np.random.choice(self.node_list)
            
            for i in range(10000):
                end_ = np.random.choice([True,False],p = [self.q,1-self.q])
                if not end_:
                    break
#                print(type(self.G.neighbors(v0)))
                connected_edges = list(self.G.neighbors(v0))
#                print(connected_edges)
                v0 = np.random.choice(connected_edges)
                
            self.G.add_edge(self.time,v0)

        self.node_list.append(self.time)
        
            
    def draw(self):
        "Draw the Network and label it."
        pos = nx.spring_layout(self.G)
        nx.draw(self.G,with_labels = True,pos = pos)

    def add_N(self,N,m):
        "Add N nodes."
        for i in trange(N):
            self.drive(m)
            
#def read_data_N(range_=10):
#    "Function to read in data for different N."
#    deg_listlists = []
#    
#    for i in range(range_):
            
            
def logbin_avg(list_list_x,list_list_y):
    "Returns the average of log bin arrays"
    unique_k = []
    new_listlistx = []
    new_listlisty = []
    
    for i in range(len(list_list_x)):
        
        curr_x = list_list_x[i]
        curr_y = list_list_y[i]
        
        for x in curr_x: 
        # check if exists in unique_list or not 
            if x not in unique_k: 
                unique_k.append(x)
            
    for i in range(len(list_list_x)):
        
        add_these = [x for x in unique_k if x not in list_list_x[i]]
        new_listx = list(list_list_x[i]) + add_these
        new_listy = list(list_list_y[i]) + list(np.zeros(len(add_these)))
        
        df = pd.DataFrame(list(zip(new_listx,new_listy)),columns=['x','y'])
        df = df.sort_values(by=['x'])
        
        new_listlistx.append(df['x'])
        new_listlisty.append(df['y'])

        
    stdx = np.std(new_listlistx,axis = 0)
    stdy = np.std(new_listlisty,axis = 0)
    
    avgx = np.mean(new_listlistx,axis = 0)
    avgy = np.mean(new_listlisty,axis = 0)
    
    
    return avgx, avgy, stdx/np.sqrt(len(new_listlistx)), \
                stdy/np.sqrt(len(new_listlistx))

def linear(x,m,c):
    y = m*x +c
    return y
    
    