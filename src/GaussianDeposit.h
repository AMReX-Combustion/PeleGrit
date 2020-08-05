#ifndef GAUSSIAN_DEPOSIT_H
#define GAUSSIAN_DEPOSIT_H

#include "Yarn.h"
#include "Dust_Pele.h"
#include "Kokkos_ScatterView.hpp"

template<int NH=0>
class GaussianDeposit {
  public:
    /* ---------------------------------------------------------------------- */
    // 3-D
    static void depositVectr3(const int NX, const int NY, const int NZ, 
                              const int NV, const int NP,
                              Yarn::VectorFieldType F, Dust_Pele::Vectr3PointType loc, 
                              Dust_Pele::HealthPointType state, Dust_Pele::Vectr3PointType P) 
    {
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));

      Kokkos::Experimental::ScatterView<double**> scatter_view(F);

      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) 
      { 
        if(state(n)!=Dust_Pele::HEALTHY) return;
        
        int ix=floor(loc(n,0));
        int jy=floor(loc(n,1));
        int kz=floor(loc(n,2));
        double rx=loc(n,0)-double(ix);
        double ry=loc(n,1)-double(jy);
        double rz=loc(n,2)-double(kz);

        double filtersum=0.0;
          
        for(int kk=-NH; kk<2+NH; kk++)
        {
          for(int jj=-NH; jj<2+NH; jj++) 
          {
            for(int ii=-NH; ii<2+NH; ii++) 
            {
              double delx=ii-rx;
              double dely=jj-ry;
              double delz=kk-rz;
              filtersum +=exp(-0.5*(delx*delx+dely*dely+delz*delz));
            } 
          }
        }

        auto F_access = scatter_view.access();

        for(int kk=-NH; kk<2+NH; kk++)
        {
          for(int jj=-NH; jj<2+NH; jj++) 
          {
            for(int ii=-NH; ii<2+NH; ii++) 
            {
              double delx=ii-rx;
              double dely=jj-ry;
              double delz=kk-rz;
              double filtercoeff=exp(-0.5*(delx*delx+dely*dely+delz*delz) )/filtersum;
              size_t nn = (kz+kk)*(NX+1)*(NY+1) + (jy+jj)*(NX+1) + (ix+ii);
              for(int l=0; l<NV; l++) F_access(nn,l)+= filtercoeff*P(n,l);
            } 
          }
        } 
      });
      Kokkos::Experimental::contribute(F,scatter_view);
    }
 
    /* ---------------------------------------------------------------------- */
    // 2-D
    static void depositVectr3(const int NX, const int NY, const int NV, const int NP,
                              Yarn::VectorFieldType F, Dust_Pele::Vectr3PointType loc, 
                              Dust_Pele::HealthPointType state, Dust_Pele::Vectr3PointType P) 
    {
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH));

      Kokkos::Experimental::ScatterView<double**> scatter_view(F);

      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) 
      { 
        if(state(n)!=Dust_Pele::HEALTHY) return;
        int ix=floor(loc(n,0));
        int jy=floor(loc(n,1));
        double rx=loc(n,0)-double(ix);
        double ry=loc(n,1)-double(jy);

        double filtersum=0.0;

        for(int jj=-NH; jj<2+NH; jj++) 
        {
          for(int ii=-NH; ii<2+NH; ii++) 
          {
            double delx=ii-rx;
            double dely=jj-ry;
            filtersum +=exp(-0.5*(delx*delx+dely*dely));
          } 
        }

        auto F_access = scatter_view.access();

        for(int jj=-NH; jj<2+NH; jj++) 
        {
          for(int ii=-NH; ii<2+NH; ii++) 
          {
            double delx=ii-rx;
            double dely=jj-ry;
            double filtercoeff=exp(-0.5*(delx*delx+dely*dely) )/filtersum;
            size_t nn = (jy+jj)*(NX+1) + (ix+ii);
            for(int l=0; l<NV; l++) F_access(nn,l)+= filtercoeff*P(n,l);
          } 
        } 
      });
      Kokkos::Experimental::contribute(F,scatter_view);
    }

    /* ---------------------------------------------------------------------- */
    // 2-D
    static void depositScalar(const int NX, const int NY, const int NP,
                              Yarn::ScalarFieldType F, Dust_Pele::Vectr3PointType loc, 
                              Dust_Pele::HealthPointType state, Dust_Pele::ScalarPointType P) 
    {
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH));

      Kokkos::Experimental::ScatterView<double*> scatter_view(F);

      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) { 
        if(state(n)!=Dust_Pele::HEALTHY) return;
        int ix=floor(loc(n,0));
        int jy=floor(loc(n,1));
        double rx=loc(n,0)-double(ix);
        double ry=loc(n,1)-double(jy);

        double filtersum=0.0;

        for(int jj=-NH; jj<2+NH; jj++) 
        {
          for(int ii=-NH; ii<2+NH; ii++) 
          {
            double delx=ii-rx;
            double dely=jj-ry;
            filtersum +=exp(-0.5*(delx*delx+dely*dely));
          } 
        }

        auto F_access = scatter_view.access();

        for(int jj=-NH; jj<2+NH; jj++) 
        {
          for(int ii=-NH; ii<2+NH; ii++) 
          {
            double delx=ii-rx;
            double dely=jj-ry;
            double filtercoeff=exp(-0.5*(delx*delx+dely*dely) )/filtersum;
            size_t nn = (jy+jj)*(NX+1) + (ix+ii);
            F_access(nn)+= filtercoeff*P(n);
          } 
        } 
      });
      Kokkos::Experimental::contribute(F,scatter_view);
    }
    /* ---------------------------------------------------------------------- */
    // 3-D
    static void depositScalar(const int NX, const int NY, const int NZ, const int NP,
                              Yarn::ScalarFieldType F, Dust_Pele::Vectr3PointType loc, 
                              Dust_Pele::HealthPointType state, Dust_Pele::ScalarPointType P) 
    {
      assert(F.dimension_0()==(NX+1+2*NH)*(NY+1+2*NH)*(NZ+1+2*NH));

      Kokkos::Experimental::ScatterView<double*> scatter_view(F);

      Kokkos::parallel_for(NP, KOKKOS_LAMBDA(const size_t& n) 
      { 
        if(state(n)!=Dust_Pele::HEALTHY) return;
        int ix=floor(loc(n,0));
        int jy=floor(loc(n,1));
        int kz=floor(loc(n,2));
        double rx=loc(n,0)-double(ix);
        double ry=loc(n,1)-double(jy);
        double rz=loc(n,2)-double(kz);

        double filtersum=0.0;

        for(int kk=-NH; kk<2+NH; kk++) 
        {
          for(int jj=-NH; jj<2+NH; jj++) 
          {
            for(int ii=-NH; ii<2+NH; ii++) 
            {
              double delx=ii-rx;
              double dely=jj-ry;
              double delz=kk-rz;
              filtersum +=exp(-0.5*(delx*delx+dely*dely+delz*delz));
            } 
          }
        }

        auto F_access = scatter_view.access();
        
        for(int kk=-NH; kk<2+NH; kk++)
        {
          for(int jj=-NH; jj<2+NH; jj++) 
          {
            for(int ii=-NH; ii<2+NH; ii++) 
            {
              double delx=ii-rx;
              double dely=jj-ry;
              double delz=kk-rz;
              double filtercoeff=exp(-0.5*(delx*delx+dely*dely+delz*delz) )/filtersum;
              size_t nn = (kz+kk)*(NX+1)*(NY+1) + (jy+jj)*(NX+1) + (ix+ii);
              F_access(nn)+= filtercoeff*P(n);
            } 
          }
        } 
      });
      Kokkos::Experimental::contribute(F,scatter_view);
    }
  };
#endif

