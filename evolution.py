import torch
from torch.autograd import Variable
from random import randint
import tensorflow as tf
import torch.nn as nn
import numpy as np
import csv
import math
import os.path
import timeit
from collections import deque
import pickle
from multiprocessing import Pool


N, D_in, H, D_out = 4361, 22, 100, 10


# = Variable(torch.zeros(N, D_in),requires_grad=False)
#y = Variable(torch.zeros(N, 3), requires_grad=False)
relu = nn.ReLU()
sig=nn.Sigmoid()
tanh=nn.Tanh()
loss_fn =nn.MSELoss()
learning_rate=0.0001


class CarData:
    
    
    def __init__(self, dataList):
        self.outputdata=[]
        self.sensordata=[]
        self.outputdata.append(dataList[0])
        self.outputdata.append(dataList[1])
        self.outputdata.append(dataList[2])
        for i in range(3,len(dataList)):
            self.sensordata.append(dataList[i])
    
    def get_output_data(self):
        return self.outputdata
    def get_sensor_data(self):
        return self.sensordata
        
    
def createCarDataList():
    
    filepath = 'aalborg.csv'  
    car_data_list=[]
    with open(filepath) as csvfile:  
        readCSV = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in readCSV:
            if(cnt!=0):
                car_data_list.append(CarData(row))
               
            
            cnt += 1
    return car_data_list

car_data_list=createCarDataList()  
print("Complete")

    

class Network:
    nueron_sizes_list=[90,100,110,120,130,140]
    hidden_layers_list=[2]
    activation_function_list=[tanh,sig]
    learning_rate_list=[0.01,0.02,0.008,0.006,0.007,0.008,.009]
    def __init__(self,p):
        self.nueron_sizes=Network.nueron_sizes_list[p[0]]
        self.hidden_layer= Network.hidden_layers_list[p[1]]
        self.activation_function=Network.activation_function_list[p[2]]
        self.dict_layers={}
        self.error=Variable(torch.zeros(1),requires_grad=False)
        self.learning_rate=Network.learning_rate_list[p[3]]
        self.p=p
        
    def create_weights(self):
        num_weights_per_layer=int(self.nueron_sizes/(self.hidden_layer+1))
        
        for i in range(0,self.hidden_layer):
            if(i==0):
                w=Variable(torch.randn(D_in, num_weights_per_layer), requires_grad=True)
                self.dict_layers[i]=w
            if(i!=0 and i<(self.hidden_layer-1)):
                w=Variable(torch.randn(num_weights_per_layer, num_weights_per_layer), requires_grad=True)
                self.dict_layers[i]=w     
            if(i!=0 and i==(self.hidden_layer-1)):
                w=Variable(torch.randn(num_weights_per_layer, 3), requires_grad=True)
                self.dict_layers[i]=w
            
            
    def forward(self,x_temp):
          

        for i in range(0,self.hidden_layer):
            
            if(i==0):
                
               
                self.output=self.activation_function(x_temp.mm(self.dict_layers[i]))
            if(i!=0 and i<(self.hidden_layer-1)):
                self.output=self.activation_function(self.output.mm(self.dict_layers[i]))
            if(i!=0 and i==(self.hidden_layer-1)):
                self.output=self.output.mm(self.dict_layers[i])

    

    def train_network(self):
        
        for epoch in range(0,100):
            self.error=Variable(torch.zeros(1),requires_grad=False)
            for i in range(0,len(car_data_list)):
                car_data=car_data_list[i]
                np_sensor_data=car_data.get_sensor_data()
                np_output_data=car_data.get_output_data()
                x_temp=Variable(torch.zeros(1,D_in),requires_grad=False)  
                y_temp=Variable(torch.zeros(1,3),requires_grad=False)  
                #print(i)
                for j in range(0,22):
                    x_temp.data[0,j]=float(np_sensor_data[j])       
                self.forward(x_temp)
                for j in range(0,3):
                    y_temp.data[0,j]=float(np_output_data[j])       
                self.forward(x_temp)
                #y_temp = Variable(torch.zeros(1, 3), requires_grad=False)
                #y_temp.data=self.output.data
                loss=loss_fn(self.output,y_temp)

                loss.backward()
                #print(loss)
                self.error+=loss
                #print(self.error)
                for keys in self.dict_layers.keys():
                    w=self.dict_layers[keys]
                    w.data=w.data-self.learning_rate*w.grad.data
                    w.grad.data.zero_()
                    self.dict_layers[keys]=w 
        print(self.error.data[0])
        print("Complete "+str(self.hidden_layer)+" "+str(self.nueron_sizes)+" "+str(self.learning_rate)+" "+str(self.p[2]))            
        return self.error


    def save_networks(networks,i):
        filename='evolution'+str(i)+'.pkl'
        with open(filename, 'wb') as output:
              pickle.dump(networks, output, pickle.HIGHEST_PROTOCOL)

    def read_networks(i):
        networks=[]
        filename='evolution'+str(i)+'.pkl'
        with open(filename, 'rb') as input:
             networks=pickle.load(input)  
        return networks        


class Mutate:
      
    def __init__(self,param_list):
        self.param_list=param_list
        self.nn={}
    def create_parent(self,param_list):
        self.total_choices=1
        for i in range(0,len(param_list)):
            self.total_choices*=(param_list[i]+1)  
        choices=[]
        for i in range(0,self.total_choices):
            p=[]
            for j in range(0,len(param_list)):
                
                p.append(randint(0, param_list[j]))
            choices.append(p)
  
        return choices

    def create_nueral_nets(self,choices):
            self.nueral_list=[]
            for i in range(0,len(choices)):
                
                network=Network(choices[i])
                network.create_weights()
                self.nn[network]=0.0
                self.nueral_list.append(network)
    def find_best_parents(self):
        for i in range (0,len(self.nueral_list)):
             error=self.nueral_list[i].train_network()
             self.nn[self.nueral_list[i]]=error
            #network.train_network()
             print(error.data[0])

param_list=[4,0,1,6]
#mutate=Mutate(param_list)
#choices=mutate.create_parent(param_list)
#mutate.create_nueral_nets(choices)
#mutate.find_best_parents()

def worker(i):
    
    p=[1,0,1,6]
    #torch.manual_seed(i*233212)
    #network=Network(p)
    network=Network.read_networks(i)[0]
    network.learning_rate=0.0009
    #network.create_weights()
    network.train_network()
    networks=[network]
    Network.save_networks(networks,i)
with Pool(3) as p:
        print(p.map(worker, [1, 2, 3,4,5,6]))
#p=[1,0,1,6]
#network=Network(p)
#network.create_weights()
#network=Network.read_networks()[0]
#network.activation_function=sig
#network.train_network()
#networks=[network]
#Network.save_networks(networks)