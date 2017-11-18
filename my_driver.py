from pytocl.driver import Driver
from pytocl.car import State, Command


# # added
# import logging
import sys
import math
import csv
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import tensorflow as tf
import torch.nn as nn
import numpy as np
import csv
import math

# _logger = logging.getLogger(__name__)



class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    ...
    # def drive(self, carstate: State) -> Command:
    #     # Interesting stuff
    #     command = Command(...)
    #     return command

    #added
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        # f = open("Test_data3.txt","a+")
        # f.write("\n*** "+str(carstate)+"\n")

        command = Command()


        
        # constructing the input 
        x_predict = [abs(carstate.speed_x)]
        x_predict.append(carstate.distance_from_center)
        x_predict.append(carstate.angle)
        [x_predict.append(i) for i in carstate.distances_from_edge]
        x_predict = np.array(x_predict)
        x_predict = x_predict.reshape(1,22)

        x_predict = scaler.transform(x_predict)

        input_sensor=tf.convert_to_tensor(data_np, np.float32)

        # predicting the output 
        y_predict = output,s_prev=predict_action( x_predict , s_prev,u_z,w_z,u_r,w_r,u_h,w_h,w_out)
        

        command.accelerator = y_predict[0][0]
        if command.accelerator < 0:
        	command.brake = y_predict[0][1]
        command.steering = y_predict[0][2]

        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1
        elif carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if not command.gear:
            command.gear = carstate.gear or 1


        
        


        return command

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
    
    filepath = 'f-speedway.csv'  
    car_data_list=[]
    with open(filepath) as csvfile:  
        readCSV = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in readCSV:
            if(cnt!=0):
                car_data_list.append(CarData(row))
               
            
            cnt += 1
    return car_data_list

N, D_in, H, D_out = 64, 22, 10, 3
dtype = torch.FloatTensor
learning_rate =0.0009
sigmoid=nn.Sigmoid()
tanh = nn.Tanh()
loss = nn.MSELoss()
#list_size=len(car_data_list)
#list_size=10
learning_rate =0.001


def read_paramters():
    if(os.path.exists('weights.txt'))==True:
        obj=torch.load('weights.txt')
        print("yes ")
        return obj.u_z,obj.w_z,obj.u_r,obj.w_r,obj.u_h,obj.w_h,obj.w_out
    else:
        u_z = Variable(torch.randn((D_in, H)),requires_grad=True)
        w_z = Variable(torch.randn((H,H)),requires_grad=True)
        u_r = Variable(torch.randn((D_in, H)),requires_grad=True)
        w_r = Variable(torch.randn((H,H)),requires_grad=True)
        u_h = Variable(torch.randn((D_in, H)),requires_grad=True)
        w_h = Variable(torch.randn((H,H)),requires_grad=True)
        w_out=Variable(torch.randn(H,3),requires_grad=True)    
        return u_z,w_z,u_r,w_r,u_h,w_h,w_out      
u_z,w_z,u_r,w_r,u_h,w_h,w_out=read_paramters()
s_prev=Variable(torch.zeros(1,H),requires_grad=False)
def predict_action(input_sensor, s_prev,u_z,w_z,u_r,w_r,u_h,w_h,w_out):
    error=Variable(torch.zeros(1),requires_grad=False)
    ones_mat=Variable(torch.ones(1,H),requires_grad=False)
    
    #car_data=car_data_list[input_data]
    #np_sensor_data=car_data.get_sensor_data()
    #print(len(np_sensor_data))
    #input_sensor=torch.Tensor(1,22)
     
    #for i in range(0,22):
     #   input_sensor[0,i]=float(np_sensor_data[i])
    #print(input_sensor) 
    input_data=Variable(input_sensor,requires_grad=False)
    z=sigmoid(input_data.mm(u_z)+s_prev.mm(w_z))
    r=sigmoid(input_data.mm(u_r)+s_prev.mm(w_r))
    h=tanh(input_data.mm(u_h)+(s_prev*r).mm(w_h))
    s=(ones_mat-z)*h+s_prev*z
    output_data=s.mm(w_out)
    return output_data,s


