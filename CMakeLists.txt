cmake_minimum_required (VERSION 3.10)
project (training)

find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-Ofast -g -fconstexpr-ops-limit=1000000000  -march=native -mtune=native -fopenmp -Wall -Wextra")

include_directories (include ../seer-nnue/include/)

add_executable(test_sample_reader test/test_sample_reader.cc)
target_link_libraries (test_sample_reader Threads::Threads)

add_executable(test_continuation test/test_continuation.cc)
target_link_libraries (test_continuation Threads::Threads)

add_executable(test_generate_links test/test_generate_links.cc)
target_link_libraries (test_generate_links Threads::Threads)

add_subdirectory(pybind11)
pybind11_add_module(seer_train src/seer_train.cc)

