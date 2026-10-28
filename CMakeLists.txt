cmake_minimum_required(VERSION 3.10)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD_REQUIRED 11)

set(SOURCES
    src/main.cpp
    src/neuron.cpp
    src/layer.cpp
    src/network.cpp
)

set(HEADERS
    include/activation.hpp
    include/neuron.hpp
    include/layer.hpp
    include/network.hpp
    include/optimizer.hpp
    include/sgd.hpp
    include/adam.hpp
)

set(CMAKE_BUILD_TYPE Debug) 
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")  

add_executable( main ${SOURCES} ${HEADERS} )
target_link_libraries( main )