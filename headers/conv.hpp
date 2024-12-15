//
// Created by Anton on 08/12/2024.
//

#ifndef CONV_HPP
#define CONV_HPP

#include "nnlayer.hpp"
#include "matrix.hpp"
#include <vector>


namespace ninf {

  class ConvolutionalLayer : NNLayer {
  public:
    ConvolutionalLayer(size_t kernelSize, size_t stepSize, size_t numLayers, double (*)(double));
    void updateWeights(const std::vector<double>&) const override;
    [[nodiscard]] Tensor3D getOutput(const Tensor3D&) const override;
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
