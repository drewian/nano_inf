#include "fnn.hpp"

#include <vector>

using namespace ninf;

FullyConnectedLayer::FullyConnectedLayer(
    const size_t noutput, const size_t ninput,
    double (*activationFunction)(double)
    )
    : noutput{noutput}, ninput{ninput}, activationFunction{activationFunction}, weights{noutput, ninput} {

}

void FullyConnectedLayer::updateWeights(const std::vector<double> &weightValues) const {
    for (size_t i = 0; i < noutput; i++) {
        for (size_t j = 0; j < ninput; j++) {
            weights.at(i, j) = weightValues.at(i * noutput + j);
        }
    }
}


Tensor3D FullyConnectedLayer::getOutput(const Tensor3D &input) const {
    // We assume a tensor with the input dimension <1, ninput, 1> (so a vector).
    // During the construction of the NN transpose layers will be injected to assure the corrected dimensions.
    Tensor3D outp{noutput, 1}; // TODO: Inefficient, outp-space could be pre-allocated during the NN-construction process.
    for (size_t i = 0; i < noutput; i++) {
        double sum = 0;
        for (size_t j = 0; j < ninput; j++) {
            sum += weights.at(i, j) * input.at(j, 0);
        }
        outp.at(i, 0) = activationFunction(sum);
    }
    return outp;
}

