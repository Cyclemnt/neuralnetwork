#include "../include/network.hpp"
#include <iostream>
#include <cmath>

Network::Network(Optimizer* optimizer_)
    : optimizer(optimizer_) {}

void Network::addLayer(int inputSize, int neuronCount, Activation activationType) {
    layers.emplace_back(inputSize, neuronCount, activationType, optimizer);
}

std::vector<float> Network::predict(const std::vector<float>& inputs) const {
    std::vector<float> activations = inputs;

    for (const auto& layer : layers)
        activations = layer.forward(activations);

    return activations;
}

void Network::train(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    int epochs, float tolerance) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalError = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<std::vector<float>> layerOutputs; // store intermediate activations
            std::vector<std::vector<float>> rawOutputs;   // store weighted sums

            std::vector<float> activation = inputs[i];
            layerOutputs.push_back(activation);

            // FORWARD PASS
            for (const auto& layer : layers) {
                rawOutputs.push_back(layer.getOutputs(activation)); // weighted sums (z)
                activation = layer.forward(activation);             // activations (a)
                layerOutputs.push_back(activation);
            }

            const auto& predicted = layerOutputs.back();
            const auto& expected = targets[i];

            // Calculation of total error
            for (size_t j = 0; j < predicted.size(); ++j)
                totalError += std::abs(expected[j] - predicted[j]);

            // BACKPROPAGATION
            std::vector<std::vector<float>> deltas(layers.size());

            // Output layer error
            size_t last = layers.size() - 1;
            deltas[last].resize(predicted.size());
            for (size_t j = 0; j < predicted.size(); ++j) {
                float z = rawOutputs[last][j];
                float derivative = ActivationFunction::derivative(z, layers[last].getActivationType());
                deltas[last][j] = (expected[j] - predicted[j]) * derivative;
            }

            // Hidden layers errors
            for (int l = last - 1; l >= 0; --l) {
                const auto& nextLayer = layers[l + 1];
                deltas[l].resize(layers[l].size());

                for (size_t j = 0; j < deltas[l].size(); ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < deltas[l + 1].size(); ++k) {
                        sum += deltas[l + 1][k] * nextLayer.getNeuron(k).getWeight(j);
                    }
                    float z = rawOutputs[l][j];
                    float derivative = ActivationFunction::derivative(z, layers[l].getActivationType());
                    deltas[l][j] = sum * derivative;
                }
            }

            // Weight update
            for (size_t l = 0; l < layers.size(); ++l) {
                layers[l].backward(layerOutputs[l], deltas[l]);
            }
        }

        if (totalError < tolerance * inputs.size()) {
            std::cout << "Training finished at epoch " << epoch << " with total error: " << totalError << std::endl;
            return;
        }
    }

    std::cout << "Training finished after " << epochs << " epochs." << std::endl;
}
