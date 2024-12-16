#ifndef FNN_HPP
#define FNN_HPP

#include "activations.hpp"
#include "nnlayer.hpp"
#include "matrix.hpp"
#include <vector>


namespace ninf {
  class FullyConnectedLayer : public NNLayer {
  public:
    FullyConnectedLayer(size_t, size_t, ACTIV_FUNC_SIG);
    void updateWeights(const std::vector<double>&) const override;
    [[nodiscard]] LayerType getLayerType() const override {
      return LayerType::FC;
    };
    [[nodiscard]] OUTPUT_FUNC override;
    void transpose(const DimsData &transposeDims) override {
      weights.transpose(transposeDims);
    }
  private:
    // A fully-connected layer is always two-dimensional.
    size_t ninput;
    size_t noutput;
    double (*activationFunction)(double);
    Tensor3D weights;
  };

}

#endif //FNN_HPP
