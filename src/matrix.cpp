#include "matrix.hpp"

#include <iterator>
#include <exception>


using namespace ninf;

Tensor3D::Tensor3D(size_t nrows, size_t ncols)
    : Tensor3D{1, nrows, ncols} {
}

Tensor3D::Tensor3D(size_t ndepth, size_t nrows, size_t ncols)
    : data{nullptr}, dims{ndepth, nrows, ncols}, transposeMapping{0, 1, 2} {
    data = new double**[ndepth];
    for (size_t i = 0; i < ndepth; i++) {
        data[i] = new double*[nrows];
        for (size_t j = 0; j < nrows; j++) {
            data[i][j] = new double[ncols];
            for (size_t k = 0; k < ncols; k++)
                data[i][j][k] = 0;
        }
    }
}

DimsData Tensor3D::getDims() const {
    return {
        dims[(transposeMapping[0] == 0) * 0 + (transposeMapping[0] == 1) * 1 + (transposeMapping[0] == 2) * 2],
        dims[(transposeMapping[1] == 0) * 0 + (transposeMapping[1] == 1) * 1 + (transposeMapping[1] == 2) * 2],
        dims[(transposeMapping[2] == 0) * 0 + (transposeMapping[2] == 1) * 1 + (transposeMapping[2] == 2) * 2]
    };
}

Tensor3D::~Tensor3D() {
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            delete[] data[i][j];
        }
    }
    delete[] data;
}

double& Tensor3D::at(const size_t i, const size_t j) const {
    return at(1, i, j);
}

double& Tensor3D::at(const size_t i, const size_t j, const size_t k) const {
    return data
        [(transposeMapping[0] == 0) * i + (transposeMapping[0] == 1) * j + (transposeMapping[0] == 2) * k]
        [(transposeMapping[1] == 0) * i + (transposeMapping[1] == 1) * j + (transposeMapping[1] == 2) * k]
        [(transposeMapping[2] == 0) * i + (transposeMapping[2] == 1) * j + (transposeMapping[2] == 2) * k];
}

void Tensor3D::transpose(const std::array<size_t, 3> &transposeMapping) {
    this->transposeMapping = transposeMapping;
}

