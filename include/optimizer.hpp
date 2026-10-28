#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

class Optimizer {
public:
    virtual void update(float& weight, float gradient) = 0;
    virtual ~Optimizer() = default;
};

#endif // OPTIMIZER_HPP
