
#include "matrix.hpp"
#include "conv.hpp"
#include <vector>

using namespace ninf;

Conv2D::Conv2D(
    const size_t kernelSize, const size_t stepSize, const size_t numLayers, double (*activationFunction)(double)
    ) : kernelSize{kernelSize}, stepSize{stepSize}, numLayers{numLayers}, activationFunction{activationFunction},
        weights{numLayers, kernelSize, kernelSize} {

}

void Conv2D::updateWeights(const std::vector<double> &weightValues) const {
    auto dims = weights.getDims();
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            for (size_t k = 0; k < dims[2]; k++) {
                weights.at(i, j, k) = weightValues.at(i * dims[0] + j * dims[1] + k);
            }
        }
    }
}

Tensor3D Conv2D::getOutput(const Tensor3D &input) {
    // Implements a basic version of Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    DimsData prevLayerDims = input.getDims();
    DimsData weightDims = weights.getDims();
    Tensor3D outp{
        numLayers,
        getNewWidthOrHeight(prevLayerDims[1], kernelSize, stepSize),
        getNewWidthOrHeight(prevLayerDims[2], kernelSize, stepSize)
    };

    // Iterate for every output filter / layer.
    for (size_t i = 0; i < weightDims[0]; i++) {
        // This is a 2D convolution, see left side of
        // https://www.researchgate.net/figure/a-2D-CONV-on-3D-input-The-filter-moves-only-in-two-directions-height-and-width-of-the_fig1_348805304
        for (size_t width = 0; width < prevLayerDims[1] - kernelSize; width += stepSize) {
            for (size_t height = 0; height < prevLayerDims[2] - kernelSize; height += stepSize) {
                double depthSum = 0;
                for (size_t depth = 0; depth < prevLayerDims[0]; depth++) {
                    double sum = 0;
                    // start iterating over the individual filter.
                    for (size_t j = 0; j < weightDims[1]; j++) {
                        for (size_t k = 0; k < weightDims[2]; k++) {
                            sum += input.at(depth, width+j, height+k) * weights.at(i, j, k);
                        }
                    }
                    depthSum += sum;
                }
                outp.at(i, width / stepSize, height / stepSize) = activationFunction(depthSum);
            }
        }
    }
    return outp;
}

