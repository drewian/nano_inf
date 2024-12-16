//
// Created by Anton on 08/12/2024.
//

#ifndef CONV_HPP
#define CONV_HPP

#include "activations.hpp"
#include "nnlayer.hpp"
#include "matrix.hpp"
#include <vector>


namespace ninf {
  inline size_t getNewWidthOrHeight(const size_t prevDim, const size_t kernelSize, const size_t stepSize) {
    return (prevDim - kernelSize) / stepSize + 1;
  }

  class Conv2D : public NNLayer {
  public:
    Conv2D(size_t kernelSize, size_t stepSize, size_t numLayers, ACTIV_FUNC_SIG);
    void updateWeights(const std::vector<double>&) const override;
    [[nodiscard]] LayerType getLayerType() const override {
      return LayerType::CONV;
    };
    [[nodiscard]] OUTPUT_FUNC override;
    void transpose(const DimsData &transposeDims) override {
      weights.transpose(transposeDims);
    }
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
