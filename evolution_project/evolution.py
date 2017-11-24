import torch
from torch.autograd import Variable
from multiprocessing import Pool
from random import randint
from mutate import Mutate
from nueral import Network
from cardata import createCarDataList
import logging


logging.basicConfig(filename='data.log',level=logging.INFO)
car_data_list=createCarDataList()
print("Complete")


networks=Network.read_networks()
def sort(networks):
    for i in range(0,len(networks)):
        temp_i=networks[i]
        for j in range(i+1,len(networks)):
            temp_j=networks[j]
            if(temp_i.error.data[0]>temp_j.error.data[0]):
                temp=temp_i
                networks[i]=networks[j]
                networks[j]=temp
    networks2=[]
    for i in range(0,10):
        networks2.append(networks[i])
    return networks2


networks=sort(networks)

def worker(temp):
    temp[0].train_network(temp[1])
    Network.save_networks(temp[0])

with Pool(6) as p:
    pool=[]
    for i in range(0,len(networks)):
        temp=[]
        temp.append(networks[i])
        temp.append(car_data_list)
        pool.append(temp)
        #networks[i].train_network(car_data_list)
    p.map(worker,pool)
     #p.map(worker, [[new_networks[0],1], [new_networks[1],2], [new_networks[2],3],[new_networks[3],4],[new_networks[4],5],[new_networks[5],6]])




#for i in range(0,100):
    #activate_networks()

#p=[1,0,1,6]
#network=Network(p)
#network.create_weights()
#network=Network.read_networks()[0]
#network.activation_function=sig
#network.train_network()
#networks=[network]
#Network.save_networks(networks)