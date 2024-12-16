//
// Created by Anton on 15/12/2024.
//

#ifndef NNLAYER_HPP
#define NNLAYER_HPP

#include "matrix.hpp"

#include <functional>
#include <vector>

#define OUTPUT_FUNC Tensor3D getOutput(const Tensor3D&)
// #define OUTPUT_FUNC_SIG Tensor3D (*)(const Tensor3D&)
#define OUTPUT_FUNC_SIG std::function<Tensor3D(Tensor3D&)>

namespace ninf {

    enum LayerType {FC, CONV};

    class NNLayer {
    public:
        virtual void updateWeights(const std::vector<double>&) const = 0;
        // TODO: Extend this function to receive a reference to a pre-allocated output tensor -> removes need for copying.
        [[nodiscard]] virtual OUTPUT_FUNC = 0;
        [[nodiscard]] virtual LayerType getLayerType() const = 0;
        virtual void transpose(const DimsData&) = 0;
        // virtual ~NNLayer() = 0;
    };

}

#endif //NNLAYER_HPP
