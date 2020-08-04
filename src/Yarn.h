#ifndef YARN_H
#define YARN_H

#include <Kokkos_Core.hpp>

class Yarn {
  private:
  public:
    typedef Kokkos::DefaultExecutionSpace::size_type size_type; 

    typedef Kokkos::View<uint64_t *>	IndexFieldType;

    typedef Kokkos::View<double *>      ScalarFieldType;
    typedef Kokkos::View<double **>     VectorFieldType;
    typedef Kokkos::View<double ***>    TensorFieldType;

    typedef Kokkos::View<double *, Kokkos::LayoutStride> StridedScalarFieldType;
};

#endif
