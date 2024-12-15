#include <iostream>
#include <vector>


#include "activations.hpp"
#include "fnn.hpp"
#include "matrix.hpp"

using namespace std;


int main()
{
    cout << "Hello World!" << endl;
    ninf::Tensor3D m({
        {1, 2, -3},
        {4, -5, -6},
        {7, 8, 9}
    });
    ninf::Tensor3D vec({
        {2},
        {2},
        {2}
    });
    ninf::Tensor3D output = relu(m * vec);

    std::vector<std::vector<double>> weightsFirstLayer = {
        {1, 2, 3},
        {4, 5, 6},
    };

    std::vector<std::vector<double>> weightsSecondLayer = {
        {0.5, 0.5},
        {0.5, 0.5},
    };

    ninf::FullyConnectedLayer fst{weightsFirstLayer.size(), weightsFirstLayer.at(0).size(), ninf::relu};
    fst.updateWeights(weightsFirstLayer);
    ninf::FullyConnectedLayer snd{weightsSecondLayer.size(), weightsSecondLayer.at(0).size(), ninf::relu};
    snd.updateWeights(weightsSecondLayer);

    auto o1 = fst.getOutput(vec);
    auto o2 = snd.getOutput(o1);

    return 0;
}
