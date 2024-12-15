add_executable(nano_inf main.cpp matrix.cpp
        fnn.cpp
        ../headers/activations.hpp
        ../headers/conv.hpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/headers)