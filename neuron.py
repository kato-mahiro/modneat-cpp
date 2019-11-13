import random
import math
from enum import Enum

try:
    from . modneat_settings import *
except:
    from modneat_settings import *

class NeuronType(Enum):
    INPUT = 1
    OUTPUT = 2
    NORMAL = 3
    MODULATION = 4
    
class Neuron:
    def __init__(self, neuron_type:NeuronType):
        self.neuron_type = neuron_type
        self.bias = random.uniform(BIAS_LOWER_LIMIT,BIAS_UPPER_LIMIT)
        self.__activation = 0.0
        self.__modulation = 0.0

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self,activation_val):
        if(self.neuron_type == NeuronType.MODULATION):
            raise Exception('ERROR: you cannot set activation val of modulation neuron')
        else:
            self.__activation = activation_val

    @property
    def modulation(self):
        return self.__modulation

    @modulation.setter
    def modulation(self,modulation_val):
        if(self.neuron_type == NeuronType.MODULATION):
            self.__modulation = modulation_val
        else:
            raise Exception('ERROR: you cannot set modulation val of not modulation neuron')

class Connetion:
    def __init__(self, connection_id, input_id, output_id):
        self.connection_id = connection_id
        self.is_valid = True
        self.weight = random.uniform(WEIGHT_LOWER_LIMIT,WEIGHT_UPPER_LIMIT)
        self.initial_weight = self.weight
        self.input_id = input_id
        self.output_id = output_id
