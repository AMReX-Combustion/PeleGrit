#ifndef _SPRAYPARTICLES_H_
#define _SPRAYPARTICLES_H_

#include <AMReX_Geometry.H>
#include "AMReX_Particles.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Amr.H"
#include <AMReX_IntVect.H>
#include <AMReX_Gpu.H>
#include "SprayFuelData.H"
#include "EOS.H"

#ifdef USE_GRIT
#include "GritPeleSprayInterface.h"
#endif

#ifdef USE_SPRAY_SOA
#define NSR_SPR 0
#define NSI_SPR 0
#define NAR_SPR AMREX_SPACEDIM+3+SPRAY_FUEL_NUM
#define NAI_SPR 0
#else
#define NSR_SPR AMREX_SPACEDIM+3+SPRAY_FUEL_NUM
#define NSI_SPR 0
#define NAR_SPR 0
#define NAI_SPR 0
#endif


class MyParIter
  : public amrex::ParIter<NSR_SPR,NSI_SPR,NAR_SPR,NAI_SPR>
{
public:
  using amrex::ParIter<NSR_SPR,NSI_SPR,NAR_SPR,NAI_SPR>::ParIter;

  std::array<RealVector, NAR_SPR>& GetAttribs() {
    return GetStructOfArrays().GetRealData();
  }

  RealVector& GetAttribs(int comp) {
    return GetStructOfArrays().GetRealData(comp);
  }
};

class MyParConstIter
  : public amrex::ParConstIter<NSR_SPR,NSI_SPR,NAR_SPR,NSI_SPR>
{
public:
  using amrex::ParConstIter<NSR_SPR,NSI_SPR,NAR_SPR,NSI_SPR>::ParConstIter;

  const std::array<RealVector, NAR_SPR>& GetAttribs() const {
    return GetStructOfArrays().GetRealData();
  }

  const RealVector& GetAttribs(int comp) const {
    return GetStructOfArrays().GetRealData(comp);
  }
};

class SprayParticleContainer
  : public amrex::AmrParticleContainer<NSR_SPR,NSI_SPR,NAR_SPR,NAI_SPR>
{
public:

  using PairIndex = std::pair<int, int>;
  using HostVectReal = amrex::Gpu::HostVector<amrex::Real>;
  using HostVectInt = amrex::Gpu::HostVector<int>;

#ifdef USE_GRIT
  GritPeleSprayInterface pele_grit;
#endif

  SprayParticleContainer(amrex::Amr* amr, amrex::BCRec* _phys_bc)
    : amrex::AmrParticleContainer<NSR_SPR,NSI_SPR,NAR_SPR,NAI_SPR>(amr),
    sub_cycle(amr->subCycle()),
    phys_bc(_phys_bc)
  {
    nstate = NSR_SPR;
    init_bcs();
#ifdef USE_GRIT
    pele_grit.initializeGrit();
#endif
  }

  ///
  /// Set fuel data
  ///
  void buildFuelData(const HostVectReal& critT,
		     const HostVectReal& boilT,
		     const HostVectReal& fuelCp,
		     const HostVectReal& fuelLatent,
		     const HostVectInt&  fuelIndx,
		     amrex::Real         ref_T)
  {
    HostVectReal fuelEnth(NUM_SPECIES);
    EOS::T2Hi(ref_T, fuelEnth.dataPtr());
    m_fuelData.build(critT, boilT, fuelCp, fuelEnth, fuelLatent, fuelIndx);
#ifdef USE_GRIT
    copyFuelData(boilT, fuelCp, fuelLatent, ref_T);
#endif
  }

#ifdef USE_GRIT
  void copyFuelData(const HostVectReal& boilT, const HostVectReal& fuelCp, 
                    const HostVectReal& fuelLatent, amrex::Real ref_T)
  {
    pele_grit.fueldata.Tref_ = ref_T;
    pele_grit.fueldata.boilT_ = boilT[0];
    pele_grit.fueldata.Cpl_ = fuelCp[0];
    pele_grit.fueldata.latentref_ = fuelLatent[0];
    pele_grit.fueldata.updateh();
  }
#endif

  ~SprayParticleContainer () 
  {
#ifdef USE_GRIT
    pele_grit.finalizeGrit();
#endif
  }

  ///
  /// Set the value of particle state for all partilces on a level
  ///
  //void SetAll (amrex::Real val, int pstate_idx, int lev);

  ///
  /// Compute a maximum time step based on the particle velocities and a particle CFL number
  ///
  amrex::Real estTimestep (int level, amrex::Real cfl) const;

  ///
  /// Update the particles but do not move particles
  ///
  void moveKick (amrex::MultiFab& state, amrex::MultiFab& source,
		 const int level, const amrex::Real& dt,
		 const amrex::Real time, const bool isVirtual,
		 const bool isGhost, const int tmp_src_width);

  ///
  /// Update and move positions of the particles
  ///
  void moveKickDrift (amrex::MultiFab& state, amrex::MultiFab& source,
		      const int level, const amrex::Real& dt,
		      const amrex::Real time, const bool isVirtual,
		      const bool isGhost, const int tmp_src_width,
		      const bool do_move, const int where_width);

  ///
  /// Update particles
  ///
  void updateParticles (const int&         lev,
			amrex::MultiFab&   state,
			amrex::MultiFab&   source,
			const amrex::Real& flow_dt,
			const amrex::Real& time,
      const int          numGhost,
			const bool         do_move);

  ///
  /// Insert particles according to some criterion
  ///
  bool insertParticles (amrex::Real time, int nstep, int lev);

  bool injectParticles (amrex::Real time, int nstep, int lev);

  void InitParticlesUniform (amrex::AmrLevel* pelec, const int& lev, const int& num_ppc);

#ifdef USE_GRIT
  void copyPeleAoStoGrit(const amrex::ArrayOfStructs<6,0>& amrparticles, const long int np,
                         const amrex::GpuArray<double, 2ul>& plo, const amrex::GpuArray<double, 2ul>& dx,
                         const amrex::Array4<const double>& statearr, const amrex::Array4<double>& sourcearr);

  void copyGrittoPeleAoS(amrex::ArrayOfStructs<6,0>& amrparticles, const long int np,
                         const amrex::GpuArray<double, 2ul>& plo, const amrex::GpuArray<double, 2ul>& dx,
                         const amrex::Array4<const double>& statearr);
#endif

private:
  int nstate;
  ///
  /// This defines reflect_lo and reflect_hi from phys_bc
  ///
  void init_bcs();

  bool sub_cycle;
  amrex::BCRec* phys_bc;
  bool reflect_lo[AMREX_SPACEDIM];
  bool reflect_hi[AMREX_SPACEDIM];
  SprayFuelData m_fuelData;
};

#endif

