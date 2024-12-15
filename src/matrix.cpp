#include "matrix.hpp"

#include <iterator>
#include <exception>


using namespace ninf;

// size_t Matrix::getRowCount() const {
//     return nrows;
// }
//
// size_t Matrix::getColCount() const {
//     return ncols;
// }

Tensor3D::Tensor3D(const std::initializer_list<std::initializer_list<double>> &inp) : data{nullptr}, dims{}, transposeMapping{0, 1, 2}
{
    data = new double**[1];
    std::initializer_list<double> row{};
    for (std::size_t j = 0; j < inp.size(); j++) {
        row = *std::next(inp.begin(), j);
        data[0] = new double*[row.size()];
        for (std::size_t k = 0; k < row.size(); k++) {
            double elem = *std::next(row.begin(), k);
            data[0][j][k] = elem;
        }
    }
    dims[0] = 1;
    dims[1] = inp.size();
    dims[2] = row.size();
}

Tensor3D::Tensor3D(const Tensor3D &m) : data{nullptr} {
    nrows = m.getRowCount();
    ncols = m.getColCount();
    copyData(m.getData(), m.getRowCount(), m.getColCount());
}

Tensor3D::Tensor3D(const TensorData mData, size_t nrows, size_t ncols) : data{nullptr}, nrows{nrows}, ncols{ncols} {
    copyData(mData, nrows, ncols);
}

Tensor3D::Tensor3D(TensorData mData, size_t nrows, size_t ncols) : data{mData}, nrows{nrows}, ncols{ncols} {
    // TODO: Verify dimensions
}

Tensor3D::Tensor3D(size_t nrows, size_t ncols) : data{nullptr}, nrows{nrows}, ncols{ncols} {
    // this constructor should be used with the set-method.
    data = new double*[nrows];
    for (size_t i = 0; i < nrows; i++)
        data[i] = new double[ncols];
}


Tensor3D& Tensor3D::operator=(const Tensor3D &m) {
    if (this == &m) // self-assignment.
        return *this;
    nrows = m.getRowCount();
    ncols = m.getColCount();
    copyData(m.getData(), m.getRowCount(), m.getColCount());
    return *this;
}

// const MatData Matrix::getData() const {
//     return const_cast<const MatData>(data);
// }

void Tensor3D::copyData(const TensorData dataToCopy, size_t nrows, size_t ncols) {
    if (data)
        delete[] data;
    data = new double*[nrows];
    for (std::size_t i = 0; i < nrows; i++) {
        data[i] = new double[ncols];
        for (std::size_t j = 0; j < ncols; j++) {
            data[i][j] = dataToCopy[i][j];
        }
    }
}

std::array<int, 3> Tensor3D::getDims() const {
    return std::array<int, 3>{
        dims[(transposeMapping[0] == 0) * 0 + (transposeMapping[0] == 1) * 1 + (transposeMapping[0] == 2) * 2],
        dims[(transposeMapping[1] == 0) * 0 + (transposeMapping[1] == 1) * 1 + (transposeMapping[1] == 2) * 2],
        dims[(transposeMapping[2] == 0) * 0 + (transposeMapping[2] == 1) * 1 + (transposeMapping[2] == 2) * 2]
    };
}

Tensor3D Tensor3D::operator*(const Tensor3D &m) const {
    auto transposedDims = getDims();
    auto mDims = m.getDims();

    Tensor3D outp({
        static_cast<size_t>(transposedDims[0]),
        static_cast<size_t>(mDims[1]),
        static_cast<size_t>(mDims[2])
    });

    for (size_t i = 0; i < transposedDims[0]; i++) {
        for (size_t j = 0; j < transposedDims[1]; j++) {
            for (size_t jj = 0; jj < mDims[1]; jj++) {
                for (size_t kk = 0; kk < mDims[2]; kk++) {
                    double sum = 0;
                    for (size_t k = 0; k < transposedDims[2]; k++)
                        sum += at(i, j, k) * m.at(k, jj, kk);
                    outp.at(i, jj, kk) = sum;
                }
            }
        }
    }

    return outp;

    // for (size_t i = 0; i < )

    // for (size_t i = 0; i < nrows; i++) {
    //     newData[i] = new double[1];
    //     double sum = 0;
    //     for (size_t j = 0; j < ncols; j++)
    //         sum += data[i][j] * m.at(j, 0);
    //     newData[i][0] = sum;
    // }
    // return Matrix{newData, nrows, 1};
}

// double Matrix::at(size_t i, size_t j) const {
//     return data[i][j];
// }

// void Matrix::set(size_t i, size_t j, double val) {
//     this->data[i][j] = val;
// }

Tensor3D::~Tensor3D() {
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
