#!/bin/bash

cmake \
-D CMAKE_BUILD_TYPE=Release \
-D GRIT_USE_HIP:BOOL=ON \
-D KOKKOS_DIR=$KOKKOS_DIR \
../src
