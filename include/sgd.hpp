#ifndef SGD_HPP
#define SGD_HPP

#include "optimizer.hpp"

class SGD : public Optimizer {
private:
    float learningRate;

public:
    explicit SGD(float lr) : learningRate(lr) {}

    void update(float& weight, float gradient) override {
        weight += learningRate * gradient;
    }
};

#endif // SGD_HPP
