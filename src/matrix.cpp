#include "matrix.hpp"

#include <iterator>
#include <exception>


using namespace ninf;

Tensor3D::Tensor3D(size_t nrows, size_t ncols)
    : Tensor3D{1, nrows, ncols} {
}

void Tensor3D::allocateData(const DimsData &dimsForAllocation) {
    data = new double**[dimsForAllocation[0]];
    for (size_t i = 0; i < dimsForAllocation[0]; i++) {
        data[i] = new double*[dimsForAllocation[1]];
        for (size_t j = 0; j < dimsForAllocation[1]; j++) {
            data[i][j] = new double[dimsForAllocation[2]];
            for (size_t k = 0; k < dimsForAllocation[2]; k++)
                data[i][j][k] = 0;
        }
    }
}

Tensor3D::Tensor3D(size_t ndepth, size_t nrows, size_t ncols)
    : data{nullptr}, dims{ndepth, nrows, ncols}, transposeMapping{0, 1, 2} {
    allocateData(dims);
}

Tensor3D::Tensor3D(const Tensor3D &t)
    : data{nullptr}, dims{t.getDims()}, transposeMapping{0, 1, 2} {
    const DimsData dimsForAllocation = getDims(); // get original dims, in case they were transposed.
    transpose(t.getTranspose());
    allocateData(dimsForAllocation);
}

Tensor3D &Tensor3D::operator=(const Tensor3D &t) {
    deleteData();
    dims = t.getDims();
    transposeMapping = {0, 1, 2};
    const DimsData dimsForAllocation = getDims();
    transposeMapping = t.getTranspose();
    allocateData(dimsForAllocation);
    return *this;
}

DimsData Tensor3D::getDims() const {
    return {
        dims[(transposeMapping[0] == 0) * 0 + (transposeMapping[0] == 1) * 1 + (transposeMapping[0] == 2) * 2],
        dims[(transposeMapping[1] == 0) * 0 + (transposeMapping[1] == 1) * 1 + (transposeMapping[1] == 2) * 2],
        dims[(transposeMapping[2] == 0) * 0 + (transposeMapping[2] == 1) * 1 + (transposeMapping[2] == 2) * 2]
    };
}

void Tensor3D::deleteData() const {
    for (size_t i = 0; i < dims[0]; i++) {
        for (size_t j = 0; j < dims[1]; j++) {
            delete[] data[i][j];
        }
        delete[] data[i];
    }
    delete[] data;
}

Tensor3D::~Tensor3D() {
    deleteData();
}

double& Tensor3D::at(const size_t i, const size_t j) const {
    return at(0, i, j);
}

double& Tensor3D::at(const size_t i, const size_t j, const size_t k) const {
    return data
        [(transposeMapping[0] == 0) * i + (transposeMapping[0] == 1) * j + (transposeMapping[0] == 2) * k]
        [(transposeMapping[1] == 0) * i + (transposeMapping[1] == 1) * j + (transposeMapping[1] == 2) * k]
        [(transposeMapping[2] == 0) * i + (transposeMapping[2] == 1) * j + (transposeMapping[2] == 2) * k];
}

void Tensor3D::transpose(const DimsData &transposeMapping) {
    this->transposeMapping = transposeMapping;
}

