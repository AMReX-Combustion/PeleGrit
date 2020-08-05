#!/bin/bash

cmake \
-D CMAKE_BUILD_TYPE=Release \
-D GRIT_USE_CUDA:BOOL=ON \
-D KOKKOS_DIR=$KOKKOS_ROOT \
../src
