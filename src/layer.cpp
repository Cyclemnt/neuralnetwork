#include "../include/layer.hpp"

Layer::Layer(int inputSize, int neuronCount, Activation activationType_, Optimizer* optimizer) 
    : activationType(activationType_) {
    for (int i = 0; i < neuronCount; ++i)
        neurons.emplace_back(inputSize, activationType, optimizer);
}

// Get raw outputs
std::vector<float> Layer::getOutputs(const std::vector<float>& inputs) const {
    std::vector<float> outputs;
    for (const auto& neuron : neurons)
        outputs.push_back(neuron.weightedSum(inputs));
    return outputs;
}

// Get outputs through activation function
std::vector<float> Layer::forward(const std::vector<float>& inputs) const {
    std::vector<float> outputs;
    for (const auto& neuron : neurons)
        outputs.push_back(neuron.forward(inputs));
    return outputs;
}

// Update weights
void Layer::backward(const std::vector<float>& inputs, const std::vector<float>& errors) {
    for (size_t i = 0; i < neurons.size(); ++i)
        neurons[i].updateWeights(inputs, errors[i]);
}

// Get layer size
int Layer::size() const {
    return neurons.size();
}

// Get neuron at index
Neuron& Layer::getNeuron(int index) {
    return neurons[index];
}

const Neuron &Layer::getNeuron(int index) const {
    return neurons[index];
}

// Get activation type
Activation Layer::getActivationType() const {
    return activationType;
}