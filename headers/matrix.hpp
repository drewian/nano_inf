#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#define TensorData double***
#define DimsData std::array<size_t, 3>

namespace ninf {
    class Tensor3D {
        // This tensor class is only meant to be used by the actual NN layers and not directly by the user of NINF.
        // The network will generate its outputs and pass the values back to the user (including activations) using
        // containers from the STL.
        // The main purpose of this class is to allow for a common interface for all layers to exist, since a 3D-Tensor
        // can contain the outputs of both a convolutional and a fully-connected layer (and also add transpose support).
    public:
        Tensor3D(size_t, size_t);
        Tensor3D(size_t, size_t, size_t);
        ~Tensor3D();
        // TODO: overload bracket operator (or maybe not, implementation could be ugly if both 2D & 3D access should be supported).
        double& at(size_t, size_t) const;
        double& at(size_t, size_t, size_t) const;
        void transpose(const std::array<size_t, 3>&);
        [[nodiscard]] DimsData getDims() const;
    private:
        TensorData data;
        DimsData dims;
        DimsData transposeMapping;
    };
}

#endif