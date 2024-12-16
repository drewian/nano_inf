#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#define ACTIV_FUNC_SIG double (*)(double)

namespace ninf {
    inline double relu(double val) {
    return val < 0 ? 0 : val;
}

}

#endif //ACTIVATIONS_HPP
