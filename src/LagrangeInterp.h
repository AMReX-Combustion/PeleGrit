#ifndef LAGRANGE_INTERP_H
#define LAGRANGE_INTERP_H

template <int NH = 0> // 2nd order
class LagrangeInterp
{
public:
  /* ---------------------------------------------------------------------- */
  // interpolation of the scalars
  // 2-D
  static void interpolateScalar(int NX, int NY, const int NP,
                                Yarn::StridedScalarFieldType F,
                                Dust_Pele::Vectr3PointType loc,
                                Dust_Pele::HealthPointType state,
                                Dust_Pele::StridedScalarPointType P)
  {
    /* ---------------------------------------------------------------------- */
    assert(F.dimension_0() == (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH));
    Kokkos::parallel_for(
        NP, KOKKOS_LAMBDA(const size_t &n) {
          if (state(n) != Dust_Pele::HEALTHY)
            return;
          int ix = floor(loc(n, 0));
          int jy = floor(loc(n, 1));

          double cx[2 * NH + 2], cy[2 * NH + 2];
          for (int i = -NH; i < 2 + NH; i++)
          {
            cx[NH + i] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != i)
                cx[NH + i] *= (loc(n, 0) - double(ix + m)) / double(i - m);
            }
          }
          for (int j = -NH; j < 2 + NH; j++)
          {
            cy[NH + j] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != j)
                cy[NH + j] *= (loc(n, 1) - double(jy + m)) / double(j - m);
            }
          }
          P(n) = 0.0;
          for (int j = -NH; j < 2 + NH; j++)
          {
            for (int i = -NH; i < 2 + NH; i++)
            {
              size_t n1 = (jy + j + NH) * (NX + 1 + 2 * NH) + ix + i + NH;
              P(n) += cx[NH + i] * cy[NH + j] * F(n1);
            }
          }
        });
  }

  /* ---------------------------------------------------------------------- */
  // 3-D
  static void interpolateScalar(int NX, int NY, int NZ, const int NP,
                                Yarn::StridedScalarFieldType F,
                                Dust_Pele::Vectr3PointType loc,
                                Dust_Pele::HealthPointType state,
                                Dust_Pele::StridedScalarPointType P)
  {
    /* ---------------------------------------------------------------------- */
    assert(F.dimension_0() ==
           (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH) * (NZ + 1 + 2 * NH));
    Kokkos::parallel_for(
        NP, KOKKOS_LAMBDA(const size_t &n) {
          if (state(n) != Dust_Pele::HEALTHY)
            return;
          int ix = floor(loc(n, 0));
          int jy = floor(loc(n, 1));
          int kz = floor(loc(n, 2));

          double cx[2 * NH + 2], cy[2 * NH + 2], cz[2 * NH + 2];
          for (int i = -NH; i < 2 + NH; i++)
          {
            cx[NH + i] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != i)
                cx[NH + i] *= (loc(n, 0) - double(ix + m)) / double(i - m);
            }
          }
          for (int j = -NH; j < 2 + NH; j++)
          {
            cy[NH + j] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != j)
                cy[NH + j] *= (loc(n, 1) - double(jy + m)) / double(j - m);
            }
          }
          for (int k = -NH; k < 2 + NH; k++)
          {
            cz[NH + k] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != k)
                cz[NH + k] *= (loc(n, 2) - double(kz + m)) / double(k - m);
            }
          }
          P(n) = 0.0;
          for (int k = -NH; k < 2 + NH; k++)
          {
            for (int j = -NH; j < 2 + NH; j++)
            {
              for (int i = -NH; i < 2 + NH; i++)
              {
                size_t n1 =
                    (kz + k + NH) * (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH) +
                    (jy + j + NH) * (NX + 1 + 2 * NH) + ix + i + NH;
                P(n) += cx[NH + i] * cy[NH + j] * cz[NH + k] * F(n1);
              }
            }
          }
        });
  }

  /* ---------------------------------------------------------------------- */
  // interpolation of vectors
  // 2-D
  static void interpolateVector(int NX, int NY, int NV, const int NP,
                                Yarn::VectorFieldType F,
                                Dust_Pele::Vectr3PointType loc,
                                Dust_Pele::HealthPointType state,
                                Dust_Pele::Vectr3PointType P)
  {
    /* ---------------------------------------------------------------------- */
    assert(F.dimension_0() == (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH) &&
           F.dimension_1() >= NV);
    Kokkos::parallel_for(
        NP, KOKKOS_LAMBDA(const size_t &n) {
          if (state(n) != Dust_Pele::HEALTHY)
            return;
          int ix = floor(loc(n, 0));
          int jy = floor(loc(n, 1));
          // here above is fine
          double cx[2 * NH + 2], cy[2 * NH + 2];
          for (int i = -NH; i < 2 + NH; i++)
          {
            cx[NH + i] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != i)
                cx[NH + i] *= (loc(n, 0) - double(ix + m)) / double(i - m);
            }
          }
          for (int j = -NH; j < 2 + NH; j++)
          {
            cy[NH + j] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != j)
                cy[NH + j] *= (loc(n, 1) - double(jy + m)) / double(j - m);
            }
          }
          for (size_t l = 0; l < NV; l++)
          {
            P(n, l) = 0.0;
            for (int j = -NH; j < 2 + NH; j++)
            {
              for (int i = -NH; i < 2 + NH; i++)
              {
                size_t n1 = (jy + j + NH) * (NX + 1 + 2 * NH) + ix + i + NH;
                assert(n1 < F.dimension_0());
                P(n, l) += cx[NH + i] * cy[NH + j] * F(n1, l);
              }
            }
          }
        });
  }

  /* ---------------------------------------------------------------------- */
  // 3-D
  static void interpolateVector(int NX, int NY, int NZ, int NV, const int NP,
                                Yarn::VectorFieldType F,
                                Dust_Pele::Vectr3PointType loc,
                                Dust_Pele::HealthPointType state,
                                Dust_Pele::Vectr3PointType P)
  {
    /* ---------------------------------------------------------------------- */
    assert(F.dimension_0() ==
               (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH) * (NZ + 1 + 2 * NH) &&
           F.dimension_1() >= NV);
    Kokkos::parallel_for(
        NP, KOKKOS_LAMBDA(const size_t &n) {
          if (state(n) != Dust_Pele::HEALTHY)
            return;
          int ix = floor(loc(n, 0));
          int jy = floor(loc(n, 1));
          int kz = floor(loc(n, 2));

          double cx[2 * NH + 2], cy[2 * NH + 2], cz[2 * NH + 2];
          for (int i = -NH; i < 2 + NH; i++)
          {
            cx[NH + i] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != i)
                cx[NH + i] *= (loc(n, 0) - double(ix + m)) / double(i - m);
            }
          }
          for (int j = -NH; j < 2 + NH; j++)
          {
            cy[NH + j] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != j)
                cy[NH + j] *= (loc(n, 1) - double(jy + m)) / double(j - m);
            }
          }
          for (int k = -NH; k < 2 + NH; k++)
          {
            cz[NH + k] = 1.0;
            for (int m = -NH; m < 2 + NH; m++)
            {
              if (m != k)
                cy[NH + k] *= (loc(n, 2) - double(kz + m)) / double(k - m);
            }
          }
          for (size_t l = 0; l < NV; l++)
          {
            P(n, l) = 0.0;
            for (int k = -NH; k < 2 + NH; k++)
            {
              for (int j = -NH; j < 2 + NH; j++)
              {
                for (int i = -NH; i < 2 + NH; i++)
                {
                  size_t n1 =
                      (kz + k + NH) * (NX + 1 + 2 * NH) * (NY + 1 + 2 * NH) +
                      (jy + j + NH) * (NX + 1 + 2 * NH) + ix + i + NH;
                  assert(n1 < F.dimension_0());
                  P(n, l) += cx[NH + i] * cy[NH + j] * cz[NH + k] * F(n1, l);
                }
              }
            }
          }
        });
  }
};

#endif
