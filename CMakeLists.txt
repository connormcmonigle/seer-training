cmake_minimum_required (VERSION 3.10)
project (training)

find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)

set (OPS_LIMIT 1000000000)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g -DNDEBUG -march=native -mtune=native -fopenmp -Wall -Wextra")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=${OPS_LIMIT}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=${OPS_LIMIT}")
endif()

include_directories (include ../seer-nnue/include/)

add_executable(test_sample_reader test/test_sample_reader.cc)
target_link_libraries (test_sample_reader Threads::Threads)

add_executable(test_continuation test/test_continuation.cc)
target_link_libraries (test_continuation Threads::Threads)

add_executable(test_generate_links test/test_generate_links.cc)
target_link_libraries (test_generate_links Threads::Threads)

add_executable(test_line_count_size test/test_line_count_size.cc)
target_link_libraries (test_generate_links Threads::Threads)

add_subdirectory(pybind11)
pybind11_add_module(seer_train src/seer_train.cc)

