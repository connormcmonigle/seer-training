cmake_minimum_required (VERSION 3.10)
project (training)

find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-Ofast -g -DNDEBUG -fconstexpr-ops-limit=1000000000  -march=native -mtune=native -fopenmp -Wall -Wextra")

include_directories (include ../seer-nnue/include/)

add_executable(test_sample_reader test/test_sample_reader.cc)

add_subdirectory(pybind11)
pybind11_add_module(seer_train src/seer_train.cc)
