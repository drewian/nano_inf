#include <iostream>
#include <fstream>
#include <vector>


#include "activations.hpp"
#include "matrix.hpp"
#include "nnbuilder.hpp"
#include "onnx.pb.h"


using namespace ninf;


onnx::GraphProto buildGraph() {
    std::ifstream input(
        R"(C:\Users\Anton\Documents\repos\notebooks\simple_model.onnx)",
        std::ios::in | std::ios::binary); // Open file
    onnx::ModelProto model;
    model.ParseFromIstream(&input); // parse file
    onnx::GraphProto graph = model.graph(); // the gragh
    return graph;
}

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
    auto graph = buildGraph();
    return 0;
}
