//
// Created by Anton on 15/12/2024.
//

#ifndef NNLAYER_HPP
#define NNLAYER_HPP

#include "matrix.hpp"

#include <vector>

namespace ninf {

    class NNLayer {
    public:
        virtual void updateWeights(const std::vector<double>&) const = 0;
        [[nodiscard]] virtual Tensor3D getOutput(const Tensor3D&) const = 0;
        virtual ~NNLayer() = 0;
    };

}

#endif //NNLAYER_HPP
