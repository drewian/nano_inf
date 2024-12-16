#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include "nnlayer.hpp"

#include <vector>
#include <memory>

namespace ninf {
    class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<OUTPUT_FUNC_SIG>&&, std::vector<std::shared_ptr<NNLayer>>&&);
        Tensor3D get(Tensor3D&);
        // getVerbose stores the layer activations for inspection purposes.
        Tensor3D getVerbose(Tensor3D&);
        [[nodiscard]] std::vector<Tensor3D> getActivations() const;

    private:
        std::vector<Tensor3D> layerActivations;
        std::vector<OUTPUT_FUNC_SIG> layerOutputFunctions;
        // Not used right now, can be used for training later.
        std::vector<std::shared_ptr<NNLayer>> layers;
    };
}

#endif //NN_HPP
