
#include "activations.hpp"
#include "conv.hpp"
#include "fnn.hpp"
#include "nnbuilder.hpp"

#include <memory>

using namespace ninf;

NNBuilder::NNBuilder(const DimsData &inputDims) : layers{}, layerFunctions{}, prevDims{inputDims} {

}

NNBuilder NNBuilder::addFCLayer(size_t noutput, size_t ninput, double (*activationFunction)(double)) {
    bool isVector = (prevDims[0] >= 1 && prevDims[1] == 1 && prevDims[2] == 1) ||
                    (prevDims[0] == 1 && prevDims[1] >= 1 && prevDims[2] == 1) ||
                    (prevDims[0] == 1 && prevDims[1] == 1 && prevDims[2] >= 1);

    if (!isVector)
        throw IllegalNetworkConstruction{};

    if (prevDims[0] != ninput && prevDims[1] != ninput && prevDims[2] != ninput)
        throw DimensionMismatch{};

    layers.push_back(
        std::make_shared<FullyConnectedLayer>(
            noutput, ninput, activationFunction
            ));

    auto layerRef = layers.back();
    DimsData dims{0, 1, 2};
    if (prevDims[0] > 1)
        dims = {1, 0, 2};
    else if (prevDims[2] > 1)
        dims = {0, 2, 1};

    layerFunctions.emplace_back([layerRef, dims](Tensor3D& t) -> Tensor3D {
        t.transpose(dims);
        return layerRef->getOutput(t);
    });

    prevDims[0] = 1;
    prevDims[1] = noutput;
    prevDims[2] = 1;

    return *this;
}


NNBuilder NNBuilder::addConv2DLayer(
    size_t kernelSize, size_t stepSize,
    size_t numLayers, double (*activationFunction) (double)
    ) {

    if (!layers.empty()) {
        if (layers.back()->getLayerType() != LayerType::CONV)
            throw IllegalNetworkConstruction{}; // 1D convolutions are currently not supported.

        if (prevDims[1] < kernelSize || prevDims[2] < kernelSize)
            throw DimensionMismatch{};
    }

    layers.push_back(std::make_shared<Conv2D>(kernelSize, stepSize, numLayers, activationFunction));
    // layerFunctions.push_back(layers.back()->getOutput);
    auto layerRef = layers.back();

    layerFunctions.emplace_back([layerRef](Tensor3D& t) -> Tensor3D {
        return layerRef->getOutput(t);
    });

    prevDims[0] = numLayers;
    prevDims[1] = getNewWidthOrHeight(prevDims[1], kernelSize, stepSize);
    prevDims[2] = getNewWidthOrHeight(prevDims[2], kernelSize, stepSize);

    return *this;
}

NNBuilder NNBuilder::updateWeightsOfPrevLayer(const std::vector<double> &weightValues) {
    layers.back()->updateWeights(weightValues);
    return *this;
}

NeuralNetwork NNBuilder::build() {
    NeuralNetwork nn{std::move(layerFunctions), std::move(layers)};
    return nn;
}
