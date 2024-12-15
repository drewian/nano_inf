#ifndef FNN_HPP
#define FNN_HPP

#include "nnlayer.hpp"
#include "matrix.hpp"
#include <vector>


namespace ninf {
class FullyConnectedLayer : NNLayer {
public:
  FullyConnectedLayer(size_t, size_t, double (*)(double));
  void updateWeights(const std::vector<double>&) const override;
  [[nodiscard]] Tensor3D getOutput(const Tensor3D&) const override;
private:
  // A fully-connected layer is always two-dimensional.
  size_t ninput;
  size_t noutput;
  double (*activationFunction)(double);
  Tensor3D weights;

};

}

#endif //FNN_HPP
