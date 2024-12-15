#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <initializer_list>
// #define std::array<std::array<double, M>, N> MatData
// #define MatData std::array<std::array<double, N>, M>
#define TensorData double***

namespace ninf {
    class Tensor3D {
        // This tensor class is only meant to be used by the actual NN layers and not directly by the user of NINF.
        // The network will generate its outputs and pass the values back to the user (including activations) using
        // containers from the STL.
        // The main purpose of this class is to allow for a common interface for all layers to exist, since a 3D-Tensor
        // can contain the outputs of both a convolutional as well as a fully-connected layer.
    public:
        // Tensor3D(const std::initializer_list<std::initializer_list<double>>&);
        // Tensor3D(const Tensor3D&);
        Tensor3D(size_t, size_t);
        Tensor3D(size_t, size_t, size_t);
        // Matrix(const MatData, size_t, size_t);
        Tensor3D(TensorData, size_t, size_t);
        // Matrix(const std::array<double, M>&);
        Tensor3D& operator=(const Tensor3D&);
        Tensor3D operator*(const Tensor3D&) const;
        // TODO: oveload bracket operator.
        ~Tensor3D();
        double& at(size_t, size_t) const;
        double& at(size_t, size_t, size_t) const;
        void transpose(const std::array<int, 3>&);
        [[nodiscard]] std::array<int, 3> getDims() const;
        // void set(size_t, size_t, double);
        // [[nodiscard]] const MatData getData() const;
        // [[nodiscard]] double at(size_t, size_t) const;
        // [[nodiscard]] size_t getRowCount() const;
        // [[nodiscard]] size_t getColCount() const;
    private:
        // double data[M][N];
        // void copyData(const MatData, size_t, size_t);
        TensorData data;
        std::array<int, 3> dims;
        std::array<int, 3> transposeMapping;
    };
}

#endif