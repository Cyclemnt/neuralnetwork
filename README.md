# Neural Network from Scratch in C++

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B11)
[![CMake](https://img.shields.io/badge/CMake-3.10%2B-064F8C.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/Dependencies-None%20(Pure%20STL)-brightgreen.svg)](#)

A lightweight, dependency-free multi-layer perceptron (MLP) implemented from scratch in modern C++.

This project was built as an educational exploration into the fundamental mechanics of artificial neural networks, backpropagation, and optimization algorithms without relying on high-level machine learning frameworks or external linear algebra libraries.

---

## Highlights

- **Pure C++ (Zero Dependencies)**: Written using solely the C++ Standard Library (`<vector>`, `<cmath>`, `<random>`, `<unordered_map>`). No BLAS, Eigen, PyTorch, or TensorFlow.
- **Fully Connected Architecture**: Modular design separating `Neuron`, `Layer`, and `Network` components with arbitrary depth and layer dimensions.
- **Custom Optimizers**:
  - **SGD** (Stochastic Gradient Descent) with customizable learning rate.
  - **Adam** (Adaptive Moment Estimation) featuring first and second moment tracking, bias correction, and numerical stabilization.
- **Activation Functions**:
  - **Sigmoid** with analytical derivative.
  - **ReLU** (Rectified Linear Unit) with piecewise derivative.
- **Backpropagation Engine**: Implementation of the generalized delta rule across hidden and output layers.
- **Early Stopping**: Automated stopping mechanism based on error tolerance thresholds.

---

## Project Structure

```text
neuralnetwork/
├── CMakeLists.txt        # CMake build configuration
├── include/
│   ├── activation.hpp    # Activation enum & functions (Sigmoid, ReLU) + derivatives
│   ├── adam.hpp          # Adam optimizer implementation with moment tracking
│   ├── layer.hpp         # Layer class managing collections of neurons
│   ├── network.hpp       # Network class orchestrating forward/backward passes
│   ├── neuron.hpp        # Individual neuron (weights, bias, forward, update)
│   ├── optimizer.hpp     # Abstract base class for optimizers
│   └── sgd.hpp           # Stochastic Gradient Descent implementation
└── src/
    ├── layer.cpp         # Layer implementation
    ├── main.cpp          # Demo application (learning binary addition)
    ├── network.cpp       # Network execution & backpropagation training loop
    └── neuron.cpp        # Neuron computation & weight initialization
```

---

## Getting Started

### Prerequisites

- A C++11 compliant compiler (GCC, Clang, or MSVC)
- CMake (version 3.10 or higher)

### Build Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Cyclemnt/neuralnetwork.git
cd neuralnetwork

# 2. Configure build with CMake
cmake -B build -S .

# 3. Build the binary
cmake --build build

# 4. Run the demo
./build/main
```

---

## Code Example

Creating and training a network is straightforward and intuitive:

```cpp
#include "network.hpp"
#include "adam.hpp"

// 1. Initialize an optimizer (Adam with default hyper-parameters)
Adam optimizer(0.001f, 0.9f, 0.999f, 1e-8f);

// 2. Build network architecture (Input: 2 -> Hidden: 16 -> Hidden: 16 -> Output: 1)
Network net(&optimizer);
net.addLayer(2, 16, Activation::RELU);
net.addLayer(16, 16, Activation::RELU);
net.addLayer(16, 1, Activation::SIGMOID);

// 3. Train on dataset (inputs, targets, max_epochs, tolerance)
net.train(inputs, targets, 100000, 0.001f);

// 4. Inference
std::vector<float> prediction = net.predict({0.5f, 0.25f});
```

---

## Demo: Learning 4-Bit Arithmetic Addition

The included [`src/main.cpp`](src/main.cpp) demonstrates the network learning arithmetic addition from scratch:

- **Problem**: Given two 4-bit integers $A, B \in [0, 15]$, predict their sum $S = A + B \in [0, 30]$.
- **Dataset**: Automatically generated and normalized ($16 \times 16 = 256$ samples).
- **Network Layout**:
  - **Inputs**: $A_{\text{norm}}, B_{\text{norm}} \in [0, 1]$ (2 inputs)
  - **Hidden Layers**: Two dense layers with 16 ReLU units each
  - **Output**: 1 unit with Sigmoid activation ($S_{\text{norm}} \in [0, 1]$)
- **Optimizer**: Adam ($\alpha = 0.001$)

The network learns the continuous linear mapping through non-linear activations solely using gradient backpropagation.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
