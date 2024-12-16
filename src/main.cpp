#include <iostream>
#include <fstream>
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
