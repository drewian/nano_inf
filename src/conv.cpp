
#include "matrix.hpp"
#include "conv.hpp"
#include <vector>

using namespace ninf;

ConvolutionalLayer::ConvolutionalLayer(
    const size_t kernelSize, const size_t stepSize, const size_t numLayers, double (*activationFunction)(double)
    ) : kernelSize{kernelSize}, stepSize{stepSize}, numLayers{numLayers}, activationFunction{activationFunction},
        weights{numLayers, kernelSize, kernelSize} {

}

void ConvolutionalLayer::updateWeights(const std::vector<double> &weightValues) const {
    auto dims = weights.getDims();
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            for (size_t k = 0; k < dims[2]; k++) {
                weights.at(i, j, k) = weightValues.at(i * dims[0] + j * dims[1] + k);
            }
        }
    }
}

Tensor3D ConvolutionalLayer::getOutput(const Tensor3D &input) const {
    DimsData prevLayerDims = input.getDims();
        // weights{numLayers, prevLayerDims[1] / stepSize - kernelSize, prevLayerDims[2] / stepSize - kernelSize} {
}

