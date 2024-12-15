#include "fnn.hpp"

#include <vector>

using namespace ninf;

FullyConnectedLayer::FullyConnectedLayer(
    const size_t noutput, const size_t ninput,
    double (*activationFunction)(double)
    )
    : noutput{noutput}, ninput{ninput}, activationFunction{activationFunction}, weights{noutput, ninput} {

}

void FullyConnectedLayer::updateWeights(const std::vector<std::vector<double> > &weightValues) {
    // TODO: Verify dimensions
    for (size_t i = 0; i < noutput; i++) {
        for (size_t j = 0; j < ninput; j++)
            weights.set(i, j, weightValues.at(i).at(j));
    }
}


Tensor3D FullyConnectedLayer::getOutput(const Tensor3D &input) const {
    auto result = (weights * input);
    const TensorData data = result.getData();
    Tensor3D output{noutput, 1};
    for (size_t i = 0; i < noutput; i++)
        output.set(i, 0, activationFunction(data[i][0]));
    return output;
}

