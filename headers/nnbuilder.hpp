#ifndef NNBUILDER_HPP
#define NNBUILDER_HPP


#include "activations.hpp"
#include "matrix.hpp"
#include "nn.hpp"
#include "nnlayer.hpp"

#include <exception>
#include <vector>
#include <memory>

namespace ninf {
    class IllegalNetworkConstruction final : std::exception {};
    class DimensionMismatch final : std::exception {};

    class NNBuilder {
    public:
        explicit NNBuilder(const DimsData&);
        NNBuilder addConv2DLayer(size_t kernelSize, size_t stepSize, size_t numLayers, ACTIV_FUNC_SIG);
        NNBuilder addFCLayer(size_t noutput, size_t ninput, ACTIV_FUNC_SIG);
        NNBuilder updateWeightsOfPrevLayer(const std::vector<double>&);
        NeuralNetwork build();
    private:
        std::vector<std::shared_ptr<NNLayer>> layers;
        // std::vector<std::function<Tensor3D(const NNLayer&, const Tensor3D&)>> layerFunctions;
        std::vector<OUTPUT_FUNC_SIG> layerFunctions;
        DimsData prevDims;
    };
}

#endif //NNBUILDER_HPP
