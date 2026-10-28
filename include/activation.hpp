#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
#include <stdexcept>

enum class Activation {
    SIGMOID,
    RELU
};

class ActivationFunction {
public:
    static float activate(float x, Activation type) {
        switch (type) {
            case Activation::SIGMOID:
                return 1.0f / (1.0f + std::exp(-x));
            case Activation::RELU:
                return x > 0.0f ? x : 0.0f;
            default:
                throw std::invalid_argument("Unsupported activation type.");
        }
    }

    static float derivative(float x, Activation type) {
        switch (type) {
            case Activation::SIGMOID: {
                float sx = activate(x, Activation::SIGMOID);
                return sx * (1.0f - sx);
            }
            case Activation::RELU:
                return x > 0.0f ? 1.0f : 0.0f;
            default:
                throw std::invalid_argument("Unsupported activation type.");
        }
    }
};

#endif // ACTIVATION_HPP
