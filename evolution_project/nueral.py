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
from random import randint
import logging
import time

relu = nn.ReLU()
sig=nn.Sigmoid()
tanh=nn.Tanh()
loss_fn =nn.MSELoss()
learning_rate=0.0001
D_in = 22
class Network:
    nueron_sizes_list = [100]
    hidden_layers_list = [2]
    activation_function_list = [sig]
    learning_rate_list = [ .009]

    def __init__(self):
        self.nueron_sizes = Network.nueron_sizes_list[0]
        self.hidden_layer = Network.hidden_layers_list[0]
        self.activation_function = Network.activation_function_list[0]
        self.dictLayers = {}
        self.error = Variable(torch.zeros(1), requires_grad=False)
        self.learning_rate = Network.learning_rate_list[0]


    def create_weights(self,random=False):
        w1=0
        w2=0
        if(random==False):
            #num_weights_per_layer = int(self.nueron_sizes / (self.hidden_layer + 1))
            w1=62
            w2=19
        else:
            w1=randint(0,80)
            w2=int((100-w1)/2)




        w = Variable(torch.randn(D_in, w1), requires_grad=True)
        self.dictLayers[0] = w

        w = Variable(torch.randn(w1, w2), requires_grad=True)
        self.dictLayers[1] = w

        w = Variable(torch.randn(w2, 3), requires_grad=True)
        self.dictLayers[2] = w

    def forward(self, x_temp):


        self.output = self.activation_function(x_temp.mm(self.dictLayers[0]))

        self.output = self.activation_function(self.output.mm(self.dictLayers[1]))

        self.output = self.output.mm(self.dictLayers[2])

    def train_network(self,car_data_list):

        for epoch in range(0, 30):
            self.error = Variable(torch.zeros(1), requires_grad=False)

            for i in range(0, len(car_data_list)):
                car_data = car_data_list[i]
                np_sensor_data = car_data.get_sensor_data()
                np_output_data = car_data.get_output_data()
                x_temp = Variable(torch.zeros(1, D_in), requires_grad=False)
                y_temp = Variable(torch.zeros(1, 3), requires_grad=False)
                # print(i)
                for j in range(0, 22):
                    x_temp.data[0, j] = float(np_sensor_data[j])
                self.forward(x_temp)
                for j in range(0, 3):
                    y_temp.data[0, j] = float(np_output_data[j])
                self.forward(x_temp)
                # y_temp = Variable(torch.zeros(1, 3), requires_grad=False)
                # y_temp.data=self.output.data
                loss = loss_fn(self.output, y_temp)

                loss.backward()
                # print(loss)
                self.error += loss
                # print(self.error)
                for keys in self.dictLayers.keys():
                    w = self.dictLayers[keys]
                    w.data = w.data - self.learning_rate * w.grad.data
                    w.grad.data.zero_()
                    self.dictLayers[keys] = w
        logging.info(self.error.data[0])
        logging.info("Complete " +str(self.dictLayers[0].data.shape)+" "+str(self.dictLayers[2].data.shape))
        return self.error

    def save_networks(network):
        filename = 'data/evolution/' + str(network.error.data[0]) + '.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)

    def read_networks():
        networks = []
        filenames=os.listdir("data/evolution")
        for filename in filenames:
            if(not filename.startswith('.')):
              file_name = 'data/evolution/' + filename
              print(filename)
              if( filename!='backup'):

                 with open(file_name, 'rb') as input:
                    network = pickle.load(input)
                    networks.append(network)
        return networks
