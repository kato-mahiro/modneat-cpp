#include "random.hpp"
#include "const.hpp"
#include<iostream>

enum NeuronType
{

    input = 1;
    output = 2;
    normal = 3;
    modulation = 4;
};

struct Neuron
{
    ~Neuron(NeuronType neuron_type)
    {
        Neuron::neuron_type = neuron_type;
        Neuron::bias = rnd(-100,100)
    }
};

int main()
{
    Neuron n;
    std
