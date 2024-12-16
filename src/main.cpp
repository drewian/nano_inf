#include <iostream>
#include <fstream>
#include <vector>


#include "activations.hpp"
#include "matrix.hpp"
#include "nnbuilder.hpp"
#include "onnx.pb.h"

#include <windows.h>


using namespace ninf;


onnx::GraphProto buildGraph() {
    std::ifstream input(
        R"(C:\Users\Anton\Documents\repos\notebooks\simple_model.onnx)",
        std::ios::in | std::ios::binary); // Open file
    onnx::ModelProto model;
    model.ParseFromIstream(&input); // parse file
    onnx::GraphProto graph = model.graph(); // the gragh

    auto inputData = *graph.input().begin();
    const auto &shape = inputData.type().tensor_type().shape();
    int dim_size = shape.dim_size();
    DimsData inputDims{1, 1, 1};
    if (dim_size == 2) // First layer is FC layer.
        inputDims[1] = shape.dim(1).dim_value();
    else { // First layer is CONV layer (or exception is thrown).
        inputDims[0] = shape.dim(1).dim_value();
        inputDims[1] = shape.dim(2).dim_value();
        inputDims[2] = shape.dim(3).dim_value();
    }

    NNBuilder nnBuilder{inputDims};

    std::vector<std::string> opTypes{};
    for (auto nodeData : graph.node()) {

        for (auto nodeInputData : nodeData.input())
            opTypes.push_back(nodeInputData);
        // for (auto nodeInputData : nodeData.input()) {
        //     std::cout << nodeInputData << std::endl;
        // }
        // auto nodeInputData = *nodeData.input().begin();
        // std::cout << "OP_TYPE: " << nodeData.op_type() << std::endl;
        // for (auto attributeData : nodeData.attribute()) {

        // }

        // auto nodeInputData = *nodeData.input().begin();
        // const auto &shape = nodeInputData.begin();
        // int dim_size = shape.dim_size();
    }

    for (int i = 1; i < dim_size; i++) { // First dim is batch_size.

    }

    // auto outp = graph.output();
    // for (auto inputData : inp) {
    //     const auto &shape = inputData.type().tensor_type().shape();
    //     int dim_size = shape.dim_size();
    //     for (int i = 1; i < dim_size; i++) {

    //     }// First dim is batch_size.
    //         std::cout << "Dim_" << i << " : " << shape.dim(i).dim_value() << std::endl;
    //     std::cout << "Name: " << inputData.name() << std::endl;
    //     for (auto dim : shape.dim()) {

    //     }
    // }
    // for (auto inputData : outp) {
    //     const auto &shape = inputData.type().tensor_type().shape();
    //     int dim_size = shape.dim_size();
    //     for (int i = 0; i < dim_size; i++)
    //         std::cout << "Dim_" << i << " : " << shape.dim(i).dim_value() << std::endl;
    //     std::cout << "Name: " << inputData.name() << std::endl;
    //     for (auto dim : shape.dim()) {

    //     }
    // }
    // for (auto node : graph.node()) {
    //     for (auto attrib : node.attribute()) {
    //         std::cout << "Name: " << attrib.name() << ",  Type: " << attrib.type() << std::endl;
    //     }
    //     // const auto &shape = valInfoData.type().tensor_type().shape();
    //     // int dim_size = shape.dim_size();
    //     // for (int i = 0; i < dim_size; i++)
    //     //     std::cout << "Dim_" << i << " : " << shape.dim(i).dim_value() << std::endl;
    //     std::cout << "---------------------------------" << std::endl;
    // }
    return graph;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
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
    buildGraph();
    return 0;
}
