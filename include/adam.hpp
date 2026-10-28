#ifndef ADAM_HPP
#define ADAM_HPP

#include "optimizer.hpp"
#include <unordered_map>
#include <cmath>

class Adam : public Optimizer {
private:
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;

    struct Moment {
        float m = 0.0f;
        float v = 0.0f;
    };

    std::unordered_map<float*, Moment> moments;
    int timestep = 0;

public:
    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    void update(float& weight, float gradient) override {
        Moment& moment = moments[&weight];

        timestep += 1;

        moment.m = beta1 * moment.m + (1.0f - beta1) * gradient;
        moment.v = beta2 * moment.v + (1.0f - beta2) * gradient * gradient;

        // Bias correction
        float mHat = moment.m / (1.0f - std::pow(beta1, timestep));
        float vHat = moment.v / (1.0f - std::pow(beta2, timestep));

        weight += learningRate * mHat / (std::sqrt(vHat) + epsilon);
    }
};

#endif // ADAM_HPP
