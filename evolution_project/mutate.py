from nueral import Network
from random import randint
import numpy as np
class Mutate:
    def __init__(self, param_list):
        self.param_list = param_list
        self.nn = {}

    def create_offspring(self,network_1,network_2):
        process=randint(0,2)
        # cross_over
        if(process==0):
           network_1,network_2=Mutate.do_cross_over(network_1,network_2)
        if(process==2):
            network_1,network_2=Mutate.do_mutate_network(network_1,network_2)
        if(process==3):
            network_1,network_2=Mutate.do_cross_over(network_1,network_2)
            network_1,network_2=Mutate.do_cross_over(network_1,network_2)
        return network_1,network_2

    def do_cross_over(network1,network2):
        network_n1=Network()
        network_n2=Network()
        network_n1.create_weights(False)
        network_n2.create_weights(False)
        split=randint(0,21)
        for i in range(0,split):
            network_n1.dictLayers[0].data[0,i]=network1.dictLayers[0].data[0,i]
            network_n2.dictLayers[0].data[0, i] = network2.dictLayers[0].data[0, i]
        for i in range(split,22):
            network_n1.dictLayers[0].data[0, i] = network2.dictLayers[0].data[0, i]
            network_n2.dictLayers[0].data[0, i] = network1.dictLayers[0].data[0, i]
        split=randint(0,19)
        for j in range(0,62):
            for i in range(0,split):
                network_n1.dictLayers[1].data[j, i] = network1.dictLayers[1].data[j, i]
                network_n2.dictLayers[1].data[j, i] = network2.dictLayers[1].data[j, i]
            for i in range(split, 19):
                network_n1.dictLayers[1].data[j, i] = network2.dictLayers[1].data[j, i]
                network_n2.dictLayers[1].data[j, i] = network1.dictLayers[1].data[j, i]
        split = randint(0,2)
        for j in range(0,19):
            for i in range(0,split):
                network_n1.dictLayers[2].data[j, i] = network1.dictLayers[2].data[j, i]
                network_n2.dictLayers[2].data[j, i] = network2.dictLayers[2].data[j, i]
            for i in range(split, 3):
                network_n1.dictLayers[2].data[j, i] = network2.dictLayers[2].data[j, i]
                network_n2.dictLayers[2].data[j, i] = network1.dictLayers[2].data[j, i]
        return network_n1,network_n2


    def do_mutate_network(network1,network2):
        layer=randint(0,2)
        if(layer==0):
          noise = np.random.normal(0, 0.2, 22*62)
          noise= np.reshape(noise,(22,62))
          network1.dictLayers[0].data=network1.dictLayers[0].data+noise
          network2.dictLayers[0].data = network1.dictLayers[0].data + noise
        if (layer == 1):
            noise = np.random.normal(0, 0.2, 62*19)
            noise = np.reshape(noise, (62, 19))
            network1.dictLayers[1].data = network1.dictLayers[1].data + noise
            network2.dictLayers[1].data = network1.dictLayers[1].data + noise
        if (layer == 2):
            noise = np.random.normal(0, 0.2, 19*3)
            noise = np.reshape(noise, (19,3))
            network1.dictLayers[2].data = network1.dictLayers[2].data + noise
            network2.dictLayers[2].data = network1.dictLayers[2].data + noise

        return network1,network2

