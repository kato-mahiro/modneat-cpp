import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

try:
    from . modneat_settings import *
    from . neuron import *
except:
    from modneat_settings import *
    from neuron import *

class NeuralNetwork:
    def __init__(self,global_max_connection_id):
        # initialize neurons
        self.neurons = []
        for n in range(INPUT_NUM):
            self.neurons.append(Neuron(NeuronType.INPUT))
        for n in range(OUTPUT_NUM):
            self.neurons.append(Neuron(NeuronType.OUTPUT))
        for n in range(NORMAL_NUM_LOWER_LIMIT):
            self.neurons.append(Neuron(NeuronType.NORMAL))
        for n in range(MODULATION_NUM_LOWER_LIMIT):
            self.neurons.append(Neuron(NeuronType.MODULATION))

        # initialize connections
        self.connections = []
        connection_id = global_max_connection_id +1
        for n in range(CONNECTION_NUM_LOWER_LIMIT):
            input_id = random.randint(0, len(self.neurons) -1)
            output_id = random.randint(INPUT_NUM, len(self.neurons) -1)
            self.connections.append(Connetion(connection_id, input_id, output_id ))
            connection_id += 1

        self.epsiron = random.uniform(EPSIRON_LOWER_LIMIT, EPSIRON_UPPER_LIMIT)

        self.fitness = 0.0

    @property
    def output_vector(self):
        output_vector = []
        for n in range(INPUT_NUM, INPUT_NUM + OUTPUT_NUM):
            output_vector.append(self.neurons[n].activation)
        return output_vector

    @property
    def local_max_connection_id(self):
        maxid = 0
        for i in range(len(self.connections)):
            if(self.connections[i].connection_id > maxid):
                maxid = self.connections[i].connection_id
        return maxid

    @property
    def num_of_normal_neuron(self):
        num = 0
        for i in range( INPUT_NUM + OUTPUT_NUM, len(self.neurons)):
            if(self.neurons[i].neuron_type == NeuronType.NORMAL):
                num += 1
        return num
    
    @property
    def num_of_modulation_neuron(self):
        num = 0
        for i in range( INPUT_NUM + OUTPUT_NUM, len(self.neurons)):
            if(self.neurons[i].neuron_type == NeuronType.MODULATION):
                num += 1
        return num

    @property
    def num_of_active_connection(self):
        num = 0
        for i in range( len(self.connections) ):
            if(self.connections[i].is_valid == True):
                num += 1
        return num

    def reset(self):
        for i in range(len(self.connections)):
            self.connections[i].weight = self.connections[i].initial_weight
        for i in range(len(self.neurons)):
            if(self.neurons[i].neuron_type != NeuronType.MODULATION):
                self.neurons[i].activation = 0.0
            else:
                self.neurons[i].modulation = 0.0
        self.fitness = 0.0

    def get_output_without_update(self,input_vector):
        if(len(input_vector) != INPUT_NUM):
            raise Exception('ERROR:num of input_vector is invalid')

        # Set input_vector
        for n in range(INPUT_NUM):
            self.neurons[n].activation = input_vector[n]


        for n in range( len(self.neurons)-1, INPUT_NUM-1, -1):
            activated_sum = 0
            modulated_sum = 0
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    activated_sum += self.neurons[self.connections[c].input_id].activation * self.connections[c].weight
                    modulated_sum += self.neurons[self.connections[c].input_id].modulation * self.connections[c].weight

            if(self.neurons[n].neuron_type != NeuronType.MODULATION):
                self.neurons[n].activation = math.tanh(activated_sum + self.neurons[n].bias)
            else:
                self.neurons[n].modulation = math.tanh(activated_sum + self.neurons[n].bias)

            # if Hebbian or ExHebbian, update weight using modulated_sum
        return self.output_vector

    def get_output_dry_run(self,input_vector):
        """
        出力ベクトルを得る
        重みを更新しない
        出力値・修飾値は0に戻る
        """
        if(len(input_vector) != INPUT_NUM):
            raise Exception('ERROR:num of input_vector is invalid')

        # Set input_vector
        for n in range(INPUT_NUM):
            self.neurons[n].activation = input_vector[n]


        for n in range( len(self.neurons)-1, INPUT_NUM-1, -1):
            activated_sum = 0
            modulated_sum = 0
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    activated_sum += self.neurons[self.connections[c].input_id].activation * self.connections[c].weight
                    modulated_sum += self.neurons[self.connections[c].input_id].modulation * self.connections[c].weight

            if(self.neurons[n].neuron_type != NeuronType.MODULATION):
                self.neurons[n].activation = math.tanh(activated_sum + self.neurons[n].bias)
            else:
                self.neurons[n].modulation = math.tanh(activated_sum + self.neurons[n].bias)

        tmp_output_vector = self.output_vector

        for i in range(len(self.neurons)):
            if(self.neurons[i].neuron_type != NeuronType.MODULATION):
                self.neurons[i].activation = 0.0
            else:
                self.neurons[i].modulation = 0.0

        return tmp_output_vector

    def show_network(self, title="no_title"):

        G = Digraph(format='png')

        for n in range(len(self.neurons)):
            if(self.neurons[n].neuron_type == NeuronType.INPUT):
                G.node( str(n), label=str(n), style='filled', fillcolor='yellow')
            elif(self.neurons[n].neuron_type == NeuronType.OUTPUT):
                labelstring = str(n) + '\n' + str(round(self.neurons[n].bias,2))
                G.node( str(n), label=labelstring, style='filled',fillcolor='turquoise' )
            elif(self.neurons[n].neuron_type == NeuronType.MODULATION):
                labelstring = str(n) + '\n' + str(round(self.neurons[n].bias,2))
                G.node( str(n), label=labelstring ,shape='square')
            elif(self.neurons[n].neuron_type == NeuronType.NORMAL):
                labelstring = str(n) + '\n' + str(round(self.neurons[n].bias,2))
                G.node( str(n), label=labelstring )

        edges = []
        edge_labels = []
        for c in range(len(self.connections)):
            if(self.connections[c].is_valid == True):
                i = self.connections[c].input_id
                o = self.connections[c].output_id
                edges.append([i,o])
                if(self.connections[c].initial_weight - self.connections[c].weight != 0):
                    edge_labels.append(str(round(self.connections[c].weight,2)) + '\n (' + str(round(self.connections[c].initial_weight - self.connections[c].weight,2)) + ')')
                else:
                    edge_labels.append(str(round(self.connections[c].weight,2)))

        for i,e in enumerate(edges):
            G.edge(str(e[0]),str(e[1]),label=edge_labels[i])

        G.view(title)
        
class HebbianNetwork(NeuralNetwork):
    def get_output_with_update(self,input_vector):
        if(len(input_vector) != INPUT_NUM):
            raise Exception('ERROR:num of input_vector is invalid')

        # Set input_vector
        for n in range(INPUT_NUM):
            self.neurons[n].activation = input_vector[n]

        for n in range( len(self.neurons)-1, INPUT_NUM-1, -1):
            activated_sum = 0
            modulated_sum = 0
            is_modulated = False
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    activated_sum += self.neurons[self.connections[c].input_id].activation * self.connections[c].weight
                    modulated_sum += self.neurons[self.connections[c].input_id].modulation * self.connections[c].weight
                    if(self.neurons[self.connections[c].input_id].neuron_type == NeuronType.MODULATION):
                        is_modulated = True

            if(self.neurons[n].neuron_type != NeuronType.MODULATION):
                self.neurons[n].activation = math.tanh(activated_sum + self.neurons[n].bias)
            else:
                self.neurons[n].modulation = math.tanh(activated_sum + self.neurons[n].bias)

            # if Hebbian or ExHebbian, update weight using modulated_sum
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    if(is_modulated == False):
                        self.connections[c].weight += \
                            self.epsiron * self.neurons[n].activation * self.neurons[ self.connections[c].input_id ].activation
                    elif(is_modulated == True):
                        self.connections[c].weight += \
                            modulated_sum * (self.epsiron * self.neurons[n].activation * self.neurons[ self.connections[c].input_id ].activation)

                    self.connections[c].weight = WEIGHT_UPPER_LIMIT if (self.connections[c].weight > WEIGHT_UPPER_LIMIT) else self.connections[c].weight
                    self.connections[c].weight = WEIGHT_LOWER_LIMIT if (self.connections[c].weight < WEIGHT_LOWER_LIMIT) else self.connections[c].weight

        return self.output_vector

class ExHebbianNetwork(NeuralNetwork):
    def __init__(self,global_max_connection_id = 0):
        super().__init__(global_max_connection_id)
        self.A= random.uniform(EVOLUTION_PARAM_LOWER_LIMIT, EVOLUTION_PARAM_UPPER_LIMIT)
        self.B= random.uniform(EVOLUTION_PARAM_LOWER_LIMIT, EVOLUTION_PARAM_UPPER_LIMIT)
        self.C= random.uniform(EVOLUTION_PARAM_LOWER_LIMIT, EVOLUTION_PARAM_UPPER_LIMIT)
        self.D= random.uniform(EVOLUTION_PARAM_LOWER_LIMIT, EVOLUTION_PARAM_UPPER_LIMIT)

    def get_output_with_update(self,input_vector):
        if(len(input_vector) != INPUT_NUM):
            raise Exception('ERROR:num of input_vector is invalid')

        # Set input_vector
        for n in range(INPUT_NUM):
            self.neurons[n].activation = input_vector[n]

        for n in range( len(self.neurons)-1, INPUT_NUM-1, -1):
            activated_sum = 0
            modulated_sum = 0
            is_modulated = False
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    activated_sum += self.neurons[self.connections[c].input_id].activation * self.connections[c].weight
                    modulated_sum += self.neurons[self.connections[c].input_id].modulation * self.connections[c].weight
                    if(self.neurons[self.connections[c].input_id].neuron_type == NeuronType.MODULATION):
                        is_modulated = True

            if(self.neurons[n].neuron_type != NeuronType.MODULATION):
                self.neurons[n].activation = math.tanh(activated_sum + self.neurons[n].bias)
            else:
                self.neurons[n].modulation = math.tanh(activated_sum + self.neurons[n].bias)

            # if Hebbian or ExHebbian, update weight using modulated_sum
            for c in range(len(self.connections)):
                if(self.connections[c].is_valid and self.connections[c].output_id == n):
                    if(is_modulated == False):
                        self.connections[c].weight += \
                            self.epsiron * \
                            (
                                self.neurons[n].activation * self.neurons[ self.connections[c].input_id ].activation * self.A + \
                                self.neurons[n].activation * self.B + \
                                self.neurons[ self.connections[c].input_id ].activation * self.C + \
                                self.D
                            )
                    elif(is_modulated == True):
                        self.connections[c].weight += \
                            modulated_sum * self.epsiron * \
                            (
                                self.neurons[n].activation * self.neurons[ self.connections[c].input_id ].activation * self.A + \
                                self.neurons[n].activation * self.B + \
                                self.neurons[ self.connections[c].input_id ].activation * self.C + \
                                self.D
                            )

                    self.connections[c].weight = WEIGHT_UPPER_LIMIT if (self.connections[c].weight > WEIGHT_UPPER_LIMIT) else self.connections[c].weight
                    self.connections[c].weight = WEIGHT_LOWER_LIMIT if (self.connections[c].weight < WEIGHT_LOWER_LIMIT) else self.connections[c].weight

        return self.output_vector
if __name__ == '__main__':
    n = HebbianNetwork(0)
    n.show_network()
    #n.get_output([1,1])
    n.show_network()
