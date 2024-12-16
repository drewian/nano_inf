add_executable(nano_inf main.cpp matrix.cpp
        fnn.cpp
        ../headers/activations.hpp
        ../headers/conv.hpp
        conv.cpp
        ../headers/NNLayer.hpp
        ../headers/nnbuilder.hpp
        ../headers/nn.hpp
        nn.cpp
        nnbuilder.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/headers)