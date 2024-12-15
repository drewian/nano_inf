#ifndef FNN_HPP
#define FNN_HPP

#include "matrix.hpp"
#include <vector>


namespace ninf {
class FullyConnectedLayer {
public:
  // We use a nested vector for the weights, and not a matrix or double ** pointer, since a vector is easier to
  // construct dynamically while parsing weight data.
  FullyConnectedLayer(size_t, size_t, double (*)(double));
  void updateWeights(const std::vector<std::vector<double>>&);
  [[nodiscard]] Tensor3D getOutput(const Tensor3D&) const;
private:
  size_t ninput;
  size_t noutput;
  double (*activationFunction)(double);
  Tensor3D weights;

};

}

#endif //FNN_HPP
