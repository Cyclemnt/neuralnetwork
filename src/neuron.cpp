#include "../include/neuron.hpp"
#include <random>

Neuron::Neuron(int inputSize, Activation activationType_, Optimizer* optimizer_)
    : activationType(activationType_), optimizer(optimizer_), biasWeight(0.0f) {

    weights.resize(inputSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0f, 1.0f); // values between -1 and 1
    for (auto& w : weights)
        w = dist(gen);
    biasWeight = dist(gen);
}

// Get raw output
float Neuron::weightedSum(const std::vector<float>& inputs) const {
    float sum = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += inputs[i] * weights[i];

    sum += bias * biasWeight;
    return sum;
}

// Get output through activation function
float Neuron::forward(const std::vector<float>& inputs) const {
    float sum = weightedSum(inputs);
    return ActivationFunction::activate(sum, activationType);
}

// Update weights
void Neuron::updateWeights(const std::vector<float>& inputs, float error) {
    float sum = weightedSum(inputs);
    float gradient = error * ActivationFunction::derivative(sum, activationType);

    for (size_t i = 0; i < weights.size(); ++i)
        optimizer->update(weights[i], gradient * inputs[i]);

    optimizer->update(biasWeight, gradient * bias);
}

// Get weight
float Neuron::getWeight(int index) const {
    return weights[index];
}