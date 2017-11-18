import torch
from torch.autograd import Variable
import tensorflow as tf
import torch.nn as nn
import numpy as np
import csv
import math
import os.path

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

car_data_list=createCarDataList()  
print("Complete")

N, D_in, H, D_out = 64, 22, 10, 3
dtype = torch.FloatTensor
learning_rate =0.0009
sigmoid=nn.Sigmoid()
tanh = nn.Tanh()
loss = nn.MSELoss()
#list_size=len(car_data_list)
#list_size=10
learning_rate =0.001

#x = Variable(torch.randn(1,25))
#y = Variable(torch.randn(1,3), requires_grad=False)


def learn_parameters(u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,start,end):

    error=Variable(torch.zeros(1),requires_grad=False)
    ones_mat=Variable(torch.ones(1,H),requires_grad=False)
    for input_data in range(start,end):
        car_data=car_data_list[input_data]
        np_sensor_data=car_data.get_sensor_data()
    #print(len(np_sensor_data))
        input_sensor=torch.Tensor(1,22)
     
        for i in range(0,22):
            input_sensor[0,i]=float(np_sensor_data[i])
    #print(input_sensor) 
        input_data=Variable(input_sensor,requires_grad=False)
        z=sigmoid(input_data.mm(u_z)+s_prev.mm(w_z))
        r=sigmoid(input_data.mm(u_r)+s_prev.mm(w_r))
        h=tanh(input_data.mm(u_h)+(s_prev*r).mm(w_h))
        s=(ones_mat-z)*h+s_prev*z
        output_data=s.mm(w_out)
    
        output_expected_data=Variable(torch.from_numpy(np.asarray(car_data.get_output_data(), dtype=np.float32)),requires_grad=False)
        output = loss(output_data, output_expected_data)
        output.backward(retain_graph=True)
        error=error+output
        u_z.data-=learning_rate*u_z.grad.data
        w_z.data -=learning_rate*w_z.grad.data
        u_r.data -=learning_rate*u_r.grad.data
        w_r.data -=learning_rate*w_r.grad.data
        u_h.data -=learning_rate*u_h.grad.data
        w_h.data -=learning_rate*w_h.grad.data
        w_out.data-=learning_rate*w_out.grad.data
    
    
        u_z.grad.data.zero_()
        w_z.grad.data.zero_()
        u_r.grad.data.zero_()
        w_r.grad.data.zero_()
        u_h.grad.data.zero_()
        w_h.grad.data.zero_()
        w_out.grad.data.zero_()
    #s_prev.grad.data.zero_()
        s_prev=s

        if(math.isnan(error.data[0])):
            print("yes")
            print(input_sensor)


    print(error)
    return u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,error

def train_model(u_z,w_z,u_r,w_r,u_h,w_h,w_out):
    errors=[]
    for i in range(0,4):
        start=0
        end=500
        s_prev=Variable(torch.zeros(1,H),requires_grad=False)
        for j in range(0,4):
             
            
            s=Variable(torch.zeros(1,H),requires_grad=True)
            u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,error=learn_parameters(u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,start,end)
            errors.append(error)
            start=start+500
            end=end+500
            print(str(j )+ " Loop Complete")
    return u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors

class WeightData:
    def __init__(self,u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors):
        self.u_z=u_z
        self.w_z=w_z
        self.u_r=u_r
        self.w_r=w_r
        self.u_h=u_h
        self.w_h=w_h
        self.s_prev=s_prev
        self.s=s
        self.w_out=w_out
        self.errors=errors
        
    def get_u_z(self):
        return self.u_z
    def get_w_z(self):
        return self.w_z
    def get_u_r(self):
        return self.u_r
    def get_w_r(self):
        return self.w_r
    def get_u_h(self):
        return self.u_h
    def get_w_h(self):
        return self.w_h
    def get_s_prev(self):
        return self.s_prev
    def get_s(self):
        return self.s
    def get_w_out(self):
        return self.w_out
    def get_error(self):
        return self.errors
    

def store_data(u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors):
    obj = WeightData(u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors)
    torch.save( obj,'weights.txt')
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
u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors=train_model(u_z,w_z,u_r,w_r,u_h,w_h,w_out)   
store_data(u_z,w_z,u_r,w_r,u_h,w_h,s_prev,s,w_out,errors)
