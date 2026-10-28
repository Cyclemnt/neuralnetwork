#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "layer.hpp"

class Network {
private:
    std::vector<Layer> layers;
    Optimizer* optimizer;

public:
    Network(Optimizer* optimizer_);

    void addLayer(int inputSize, int neuronCount, Activation activationType = Activation::SIGMOID);

    std::vector<float> predict(const std::vector<float>& inputs) const;

    void train(const std::vector<std::vector<float>>& inputs,
               const std::vector<std::vector<float>>& targets,
               int epochs = 10000, float tolerance = 0.01f);
};

#endif // NETWORK_HPP
