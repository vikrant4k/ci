import torch
from torch.autograd import Variable
from multiprocessing import Pool
from random import randint
from mutate import Mutate
from my_driver import MyDriver
from nueral import Network
from my_driver import createCarDataList

# mutate=Mutate(param_list)
# choices=mutate.create_parent(param_list)
# mutate.create_nueral_nets(choices)
# mutate.find_best_parents()
def worker(arr):
    p = [1, 0, 1, 6]
    network = arr[0]
    # torch.manual_seed(i*233212)
    # network=Network(p)
    # network=networks[i]
    network.learning_rate = 0.009

    # network.create_weights()
    network.train_network()
    networks = [network]
    Network.save_networks(networks, arr[1])


def create_new_weight(w1, w2):
    w_new = Variable(torch.randn(w1.data.shape), requires_grad=True)
    w_new_1 = Variable(torch.randn(w1.data.shape), requires_grad=True)
    split = randint(0, w1.data.shape[1] - 1)

    for i in range(0, w1.data.shape[0]):
        for j in range(0, split):
            w_new.data[i, j] = w1.data[i, j]
            w_new_1.data[i, j] = w2.data[i, j]
        for j in range(split, w1.data.shape[1]):
            w_new.data[i, j] = w2.data[i, j]
            w_new_1.data[i, j] = w1.data[i, j]
    return w_new, w_new_1


def activate_networks():
    p = [1, 0, 1, 6]
    networks = []
    new_networks = []
    for i in range(1, 7):
        network = Network.read_networks(i)[0]
        networks.append(network)
    for i in range(0, 6):
        temp_i_network = networks[i]
        for j in range(i, 6):
            temp_j_network = networks[j]
            if (temp_i_network.error.data[0] > temp_j_network.error.data[0]):
                temp_network = networks[i]
                networks[i] = networks[j]
                networks[j] = temp_network

    for i in range(0, 3):
        network = Network(p)
        network.create_weights()
        network_1 = Network(p)
        network_1.create_weights()
        parent1_index = randint(0, 4)
        parent2_index = randint(0, 4)
        w1 = networks[parent1_index].dict_layers[0]
        w2 = networks[parent2_index].dict_layers[0]
        w_new, w_new_1 = create_new_weight(w1, w2)
        network.dict_layers[0] = w_new
        network_1.dict_layers[0] = w_new_1
        w1 = networks[parent1_index].dict_layers[1]
        w2 = networks[parent2_index].dict_layers[1]
        w_new, w_new_1 = create_new_weight(w1, w2)
        network.dict_layers[1] = w_new
        network_1.dict_layers[1] = w_new_1
        new_networks.append(network)
        new_networks.append(network_1)
    print(len(new_networks))
    with Pool(6) as p:

        p.map(worker, [[new_networks[0], 1], [new_networks[1], 2], [new_networks[2], 3], [new_networks[3], 4],
                       [new_networks[4], 5], [new_networks[5], 6]])


for i in range(0, 100):
    activate_networks()

# p=[1,0,1,6]
# network=Network(p)
# network.create_weights()
# network=Network.read_networks()[0]
# network.activation_function=sig
# network.train_network()
# networks=[network]
