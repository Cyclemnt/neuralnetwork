#ifndef LAYER_HPP
#define LAYER_HPP

#include "neuron.hpp"

class Layer {
private:
    std::vector<Neuron> neurons;
    Activation activationType;

public:
    Layer(int inputSize, int neuronCount, Activation activationType_, Optimizer* optimizer);

    std::vector<float> getOutputs(const std::vector<float>& inputs) const;
    std::vector<float> forward(const std::vector<float>& inputs) const;

    void backward(const std::vector<float>& inputs, const std::vector<float>& errors);

    int size() const;
    Neuron& getNeuron(int index);
    const Neuron& getNeuron(int index) const;
    Activation getActivationType() const;
};

#endif // LAYER_HPP
