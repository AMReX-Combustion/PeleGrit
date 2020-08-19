#!/bin/sh
KOKKOS_VERSION=3.1.00
#KOKKOS_OPTION="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_OPTION=ON"
KOKKOS_OPTIONS="-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_PASCAL61=ON"
KOKKOS_DIR=/opt/sw/kokkos.${KOKKOS_VERSION}.cuda
wget --quiet https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz
mkdir -p kokkos
tar -xf ${KOKKOS_VERSION}.tar.gz -C kokkos --strip-components=1
cd kokkos
mkdir -p build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${KOKKOS_DIR} -D CMAKE_CXX_COMPILER=$PWD/../bin/nvcc_wrapper ${KOKKOS_OPTIONS} ..
make -j16 install
