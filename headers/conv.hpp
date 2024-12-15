//
// Created by Anton on 08/12/2024.
//

#ifndef CONV_HPP
#define CONV_HPP

#include "matrix.hpp"
#include <vector>


namespace ninf {

  class ConvolutionalLayer {
  public:
    ConvolutionalLayer(size_t kernelSize, size_t stepSize, size_t numLayers, double (*)(double));
    void updateWeights(const std::vector<double>&) const;
    [[nodiscard]] Tensor3D getOutput(const Tensor3D&) const;
  private:
    // A fully-connected layer is always two-dimensional.
    size_t kernelSize;
    size_t stepSize;
    size_t numLayers;
    double (*activationFunction)(double);
    Tensor3D weights;
  };

}

#endif //CONV_HPP
