#ifndef NEURON_HPP
#define NEURON_HPP

#include "activation.hpp"
#include "optimizer.hpp"
#include <vector>

class Neuron {
private:
   std::vector<float> weights;
    float biasWeight;
    float bias = 1.0f;

    Activation activationType;
    Optimizer* optimizer;

public:
    Neuron(int inputSize, Activation activationType_, Optimizer* optimizer_);

    float weightedSum(const std::vector<float>& inputs) const;
    float forward(const std::vector<float>& inputs) const;

    void updateWeights(const std::vector<float>& inputs, float error);

    float getWeight(int index) const;
};

#endif // NEURON_HPP