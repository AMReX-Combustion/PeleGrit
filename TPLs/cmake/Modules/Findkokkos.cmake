# module for kokkos
include(FindPackageHandleStandardArgs)

IF(NOT KOKKOS_DIR)
    IF(GRIT_USE_CUDA)   
	SET(KOKKOS_DIR "${CMAKE_SOURCE_DIR}/../TPLs/kokkos")
    ELSE()
	SET(KOKKOS_DIR "${CMAKE_SOURCE_DIR}/../TPLs/kokkosOMP")
    ENDIF()
ENDIF()

find_path(
    KOKKOS_INCLUDE_DIR Kokkos_Core.hpp
    HINTS ${KOKKOS_DIR}
    PATH_SUFFIXES include
)

find_library(
    KOKKOS_LIBRARY 
    NAMES kokkos
    HINTS ${KOKKOS_DIR}
    PATH_SUFFIXES kokkos lib 
)

find_package_handle_standard_args(
    kokkos  DEFAULT_MSG
    KOKKOS_LIBRARY KOKKOS_INCLUDE_DIR
)

IF(KOKKOS_FOUND)
  SET(KOKKOS_LIBRARIES
      ${KOKKOS_LIBRARY}
  )
  SET(KOKKOS_INCLUDE_DIRS
      ${KOKKOS_INCLUDE_DIR}
  )
  mark_as_advanced(KOKKOS_INCLUDE_DIRS KOKKOS_LIBRARIES )
ELSE()
  SET(KOKKOS_DIR "" CACHE PATH
    "An optional hint to the kokkos installation directory"
    )
ENDIF()
