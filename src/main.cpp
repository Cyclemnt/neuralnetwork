#include "../include/network.hpp"
#include "../include/adam.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

// Function to generate the normalized addition dataset on n_bits
void generateDataset(int n_bits, 
                     std::vector<std::vector<float>>& inputs, 
                     std::vector<std::vector<float>>& targets) {
    int max_val = (1 << n_bits) - 1;  // max value for n_bits bits
    int max_sum = 2 * max_val;        // max possible sum

    inputs.clear();
    targets.clear();

    for (int A = 0; A <= max_val; ++A) {
        for (int B = 0; B <= max_val; ++B) {
            float A_norm = static_cast<float>(A) / max_val;
            float B_norm = static_cast<float>(B) / max_val;
            float S = A + B;
            float S_norm = S / max_sum;

            inputs.push_back({A_norm, B_norm});
            targets.push_back({S_norm});
        }
    }
}

int main() {
    Adam optimizer(0.001f, 0.9f, 0.999f, 1.0E-8F);
    Network net(&optimizer);
    net.addLayer(2, 16, Activation::RELU);
    net.addLayer(16, 16, Activation::RELU);
    net.addLayer(16, 1, Activation::SIGMOID);

    // Automatic generation of the 4-bit dataset
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
    generateDataset(4, inputs, targets);
    std::cout << "Dataset size: " << inputs.size() << " samples\n";

    net.train(inputs, targets, 100000, 0.001f);

    for (const auto& input : inputs) {
        auto prediction = net.predict(input);
        float A = input[0] * 15;  // 15 = max_val for 4 bits
        float B = input[1] * 15;
        float S_pred = prediction[0] * 30; // max_sum = 30
        std::cout << std::fixed << std::setprecision(2)
                  << A << " + " << B << " = " << S_pred << std::endl;
    }
    
    return 0;
}