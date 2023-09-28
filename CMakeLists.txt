cmake_minimum_required (VERSION 3.10)
project (training)
find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)

set (OPS_LIMIT 1000000000)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -flto -g -DNDEBUG -march=native -mtune=native -fopenmp -Wall -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconstexpr-ops-limit=${OPS_LIMIT}")


message (STATUS "EVALFILE=${EVALFILE}")
add_compile_definitions (EVALFILE="${EVALFILE}")

include_directories (include seer-nnue/include/ seer-nnue/incbin/ seer-nnue/syzygy/)

file(GLOB CHESS_SRC_FILES seer-nnue/src/chess/*.cc)
file(GLOB SEARCH_SRC_FILES seer-nnue/src/search/*.cc)


add_subdirectory (pybind11)
pybind11_add_module (
  seer_train
  src/seer_train.cc

  ${CHESS_SRC_FILES}
  ${SEARCH_SRC_FILES}
  seer-nnue/syzygy/tbprobe.cc
)

