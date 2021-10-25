cmake_minimum_required (VERSION 3.10)
project (training)

find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)

set (OPS_LIMIT 1000000000)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -flto -g -DNDEBUG -march=native -mtune=native -fopenmp -Wall -Wextra")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=${OPS_LIMIT}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-steps=${OPS_LIMIT}")
endif()

add_compile_definitions(EVALFILE="eval.bin")

include_directories (include ../seer-nnue/include/)
include_directories (include ../seer-nnue/incbin/)
include_directories (include ../seer-nnue/syzygy/)


add_executable(test_sample_reader test/test_sample_reader.cc ../seer-nnue/syzygy/tbchess.c ../seer-nnue/syzygy/tbprobe.c)
target_link_libraries (test_sample_reader Threads::Threads)

add_executable(test_data_gen test/test_data_gen.cc ../seer-nnue/syzygy/tbchess.c ../seer-nnue/syzygy/tbprobe.c)
target_link_libraries (test_data_gen Threads::Threads)

add_subdirectory(pybind11)
pybind11_add_module(seer_train src/seer_train.cc ../seer-nnue/syzygy/tbchess.c ../seer-nnue/syzygy/tbprobe.c)

