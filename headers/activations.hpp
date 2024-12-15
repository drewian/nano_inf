//
// Created by Anton on 08/12/2024.
//

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

namespace ninf {

double relu(double val) {
    return val < 0 ? 0 : val;
}

}

#endif //ACTIVATIONS_HPP
