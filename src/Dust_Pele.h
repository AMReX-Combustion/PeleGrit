#ifndef DUST_PELE_H
#define DUST_PELE_H

#ifndef BLOCKNDUST
#define BLOCKNDUST 10000
#endif

#include <Kokkos_Core.hpp>

#include <cstdint>

class Dust_Pele
{
public:
  static const size_t NDUST = BLOCKNDUST;
  enum STATE : uint32_t;
  using st_type = typename std::underlying_type<STATE>::type;

  typedef Kokkos::View<double[NDUST]> ScalarPointType;
  typedef Kokkos::View<double[NDUST][3]> Vectr3PointType;
  typedef Kokkos::View<double[NDUST], Kokkos::LayoutStride>
      StridedScalarPointType;
  typedef Kokkos::View<st_type[NDUST]> HealthPointType;
  typedef Kokkos::View<uint64_t[NDUST]> IndexPointType;

  typedef Kokkos::View<const double[NDUST]> ConstScalarPointType;
  typedef Kokkos::View<const double[NDUST][3]> ConstVectr3PointType;

  IndexPointType ssn;
  HealthPointType state;
  Vectr3PointType loc;
  Vectr3PointType vel;
  ScalarPointType dia;
  ScalarPointType Re;

  enum STATE : uint32_t
  {
    UNOCCUPIED = 0, //
    HEALTHY = 1,    //
    WENT_TOO_FAR    // !Went farther than the nearest neighbor
  };

public:
  // constructor
  Dust_Pele()
  {
    ssn = IndexPointType("ssn");
    state = HealthPointType("state");
    loc = Vectr3PointType("loc");
    vel = Vectr3PointType("vel");
    dia = ScalarPointType("dia");
    Re = ScalarPointType("Re");
  }

  void initialize(uint64_t ssn_start)
  {
    Kokkos::parallel_for(NDUST, init_ssn(ssn, ssn_start));
  }
  /* ---------------------------------------------------------------------- */
  struct init_ssn
  {
    uint64_t ssn_start;
    IndexPointType ssn;
    // constructor
    init_ssn(Kokkos::View<uint64_t[NDUST]> ssn_, uint64_t ssn_start_)
        : ssn(ssn_)
        , ssn_start(ssn_start_){};
    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t n) const { ssn(n) = ssn_start + n; }
  };
  /* ---------------------------------------------------------------------- */
  int getcount(STATE s = HEALTHY) const
  {
    int nfilled;
    size_t ND = NDUST;
    HealthPointType state_ = state;
    Kokkos::parallel_reduce(
        ND,
        KOKKOS_LAMBDA(const size_t &n, int &count) {
          if (state_(n) == s)
            count++;
        },
        nfilled);
    return (nfilled);
  }
};
#endif
