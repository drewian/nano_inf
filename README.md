# nano_inf

nano_inf is a lightweight inference framework for deep learning projects.
It is designed for simplicity and mainly written for learning purposes.
The framework currently supports fully-connected and convolutional layers and operates entirely on the CPU, without requiring any external dependencies.

## Features

- **Simple API**: Build your neural networks using a flexible Builder pattern.
- **Compact Representation**: The network is represented as a vector of function pointers for efficient execution.
- **No External Dependencies**: Runs entirely on the CPU without requiring additional libraries or frameworks.
- **Expandable**: Planned support for ONNX format (see Roadmap), allowing networks trained in frameworks like PyTorch to be imported and executed.

## Getting Started

### Prerequisites

There are no external dependencies required for nano_inf. The framework is entirely self-contained.

### Usage

#### Defining a Network

Use the Builder pattern to define your network architecture:

```cpp
#include <vector>

#include "activations.hpp"
#include "matrix.hpp"
#include "nnbuilder.hpp"

using namespace ninf;

int main() {

    NNBuilder nnBuilder{DimsData {3, 100, 100}};
    NeuralNetwork nn = nnBuilder
        .addConv2DLayer(10, 10, 10, relu)
        .addConv2DLayer(10, 1, 50, relu)
        .addFCLayer(10, 50, relu)
        .addFCLayer(3, 10, relu)
        .build();

    auto inp = Tensor3D{3, 100, 100};
    auto outp = nn.get(inp);
    outp = nn.getVerbose(inp);
    auto activations = nn.getActivations();
    return 0;
}

```

## Roadmap

- **ONNX Support**: Add support for loading ONNX models, enabling compatibility with popular frameworks like PyTorch and TensorFlow.
- **Additional Layers**: Expand support for advanced layer types (e.g., pooling, batch normalization).
- **Optimizations**: Improve performance, memory for layer output can be pre-allocated.

## Contributing / License

Feel free to use this code however you please, but I will likely not accept any contributions, since this project is mainly for self-study.
Suggestions are always welcome though!

## Contact

For questions or feedback, feel free to contact [anton.drewing@gmail.com] or open an issue on GitHub.
