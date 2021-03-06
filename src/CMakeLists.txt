cmake_minimum_required (VERSION 2.8)

project (GRIT CXX)

#----------------------------------------------------------------------
set(PROJECT_TPL_DIR ${CMAKE_SOURCE_DIR}/../TPLs)

#----------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_TPL_DIR}/cmake/Modules)
find_package(kokkos REQUIRED)
include_directories(${KOKKOS_INCLUDE_DIRS})

option(GRIT_USE_CUDA "Is it a CUDA build?" OFF)

if(GRIT_USE_CUDA)
	set(CMAKE_CXX_COMPILER ${KOKKOS_DIR}/bin/nvcc_wrapper)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
endif()

if(GRIT_USE_HIP)
	set(CMAKE_CXX_COMPILER hipcc)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-gpu-rdc --amdgpu-target=gfx906,gfx908")
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp")
endif()

#----------------------------------------------------------------------
set(HOSTNAME $ENV{HOSTNAME})
if(NOT HOSTNAME)
  SITE_NAME(HOSTNAME)
endif()

if(${HOSTNAME} MATCHES "^summitdev")
  set(SERIALEXEC "jsrun")
  set(SERIALEXEC_PREFLAGS "-n1 -g1")
  set(MPIEXEC "jsrun -n1 -g1 -c ALL_CPUS")
  set(MPIEXEC_NUMPROC_FLAG "-a")
elseif(${HOSTNAME} MATCHES "^summit")
  set(SERIALEXEC "jsrun")
  set(SERIALEXEC_PREFLAGS -n1 -g1 --smpiargs none)
  set(MPIEXEC "jsrun")
  set(MPIEXEC_PREFLAGS -n1 -g1 -c ALL_CPUS)
  set(MPIEXEC_NUMPROC_FLAG "-a")
elseif(${HOSTNAME} MATCHES "^percival")
  set(SERIALEXEC "aprun")
  set(MPIEXEC "aprun")
  set(MPIEXEC_NUMPROC_FLAG "-n")
  set(LAPACK_LIBRARIES "") #Compiler brings this in
elseif(${HOSTNAME} MATCHES "^titan")
  set(SERIALEXEC "aprun")
  set(MPIEXEC "aprun")
  set(MPIEXEC_NUMPROC_FLAG "-n")
  set(LAPACK_LIBRARIES "") #Compiler brings this in
endif()

#----------------------------------------------------------------------
set(LIBRARIESLIST ${KOKKOS_LIBRARIES} ${MPI_CXX_LIBRARIES} ${CMAKE_DL_LIBS})

include_directories(${CMAKE_SOURCE_DIR})

#----------------------------------------------------------------------
message(STATUS "HOSTNAME is " ${HOSTNAME})
message(STATUS "CMAKE_BUILD_TYPE is " ${CMAKE_BUILD_TYPE})

#-----------------------------------------------------------------------
add_library(gritpelesprayinterface ${CMAKE_SOURCE_DIR}/GritPeleSprayInterface.cpp)
set_property(TARGET gritpelesprayinterface APPEND PROPERTY COMPILE_DEFINITIONS BLOCKNDUST=4800)

add_executable(ParallelforTest ParallelforTest.cpp)
target_link_libraries(ParallelforTest ${LIBRARIESLIST})

