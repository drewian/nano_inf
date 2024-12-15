
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
    // This is a 3D convolution, see right side of
    // https://www.researchgate.net/figure/a-2D-CONV-on-3D-input-The-filter-moves-only-in-two-directions-height-and-width-of-the_fig1_348805304
    // for an illustration.
    DimsData prevLayerDims = input.getDims();
    DimsData weightDims = weights.getDims();
    Tensor3D outp{numLayers, prevLayerDims[1] / stepSize - kernelSize, prevLayerDims[2] / stepSize - kernelSize};
    for (size_t depth = 0; depth < prevLayerDims[0]; depth+= stepSize) {
        for (size_t width = 0; width < prevLayerDims[1]; width += stepSize) {
            for (size_t height = 0; height < prevLayerDims[2]; height += stepSize) {
                double sum = 0;
                // start iterating over the weight-tensor.
                for (size_t i = 0; i < weightDims[0]; i++) {
                    for (size_t j = 0; j < weightDims[1]; j++) {
                        for (size_t k = 0; k < weightDims[2]; k++) {
                            sum += input.at(depth+i, width+j, height+k) * weights.at(i, j, k);
                        }
                    }
                }
                outp.at(depth, width, height) = activationFunction(sum);
            }
        }
    }
    return outp;
}

