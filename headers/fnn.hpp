#ifndef FNN_HPP
#define FNN_HPP

#include "matrix.hpp"
#include <vector>


namespace ninf {
class FullyConnectedLayer {
public:
  FullyConnectedLayer(size_t, size_t, double (*)(double));
  void updateWeights(const std::vector<double>&) const;
  [[nodiscard]] Tensor3D getOutput(const Tensor3D&) const;
private:
  // A fully-connected layer is always two-dimensional.
  size_t ninput;
  size_t noutput;
  double (*activationFunction)(double);
  Tensor3D weights;

};

}

#endif //FNN_HPP
