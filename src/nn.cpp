
#include "matrix.hpp"
#include "nn.hpp"

using namespace ninf;


NeuralNetwork::NeuralNetwork(std::vector<OUTPUT_FUNC_SIG> &&outputFunctions, std::vector<std::shared_ptr<NNLayer>> &&layers)
    : layerOutputFunctions{outputFunctions}, layers{layers} {

}

Tensor3D NeuralNetwork::get(Tensor3D& input) {
    auto outp = layerOutputFunctions.at(0)(input);
    for (size_t i = 1; i < layerOutputFunctions.size(); i++) {
        outp = layerOutputFunctions.at(i)(outp);
    }
    return outp;
}

Tensor3D NeuralNetwork::getVerbose(Tensor3D& input) {
    layerActivations = std::vector<Tensor3D>{};
    auto outp = layerOutputFunctions.at(0)(input);
    for (size_t i = 1; i < layerOutputFunctions.size(); i++) {
        layerActivations.push_back(outp);
        outp = layerOutputFunctions.at(i)(outp);
    }
    return outp;
}

std::vector<Tensor3D> NeuralNetwork::getActivations() const {
    return layerActivations;
}
