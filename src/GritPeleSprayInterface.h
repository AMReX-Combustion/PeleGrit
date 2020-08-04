#ifndef GRIT_PELE_SPRAY_INTERFACE_H
#define GRIT_PELE_SPRAY_INTERFACE_H

//#if !defined(BLOCKNX) || !defined(BLOCKNY) || !defined(BLOCKNZ)
//#error BLOCKNX, BLOCKNY and BLOCKNZ have to be defined
//#endif

#include <memory>
#include "PeleSprayParticleClass.h"
#include "Fuel.h"

class GritPeleSprayInterface {
  private:
  public:
    std::unique_ptr<PeleSprayParticle[]> ptr_spray;
    Fuel fueldata;
    const double* ptr_state=nullptr;
    double* ptr_source=nullptr;

    GritPeleSprayInterface(){
    };

    ~GritPeleSprayInterface(){
    }

    void initializeGrit();
    void finalizeGrit();
    void advanceSprays(const long int np, const double dt, 
                       const double* dx, const int DIM,
                       const int NX, const int NY, const int NZ,
                       const int rhogindex, const int rhogUgindex,
                       const int rhogegindex, const int Tgindex, 
                       const int Ygindex, bool do_move,
                       const int sNX, const int sNY, const int sNZ);
    int getIndex(const int small_n, const int small_NX,
                       const int small_NY, const int small_NZ,
                       const int DIFFNX, const int DIFFNY,
                       const int DIFFNZ);
};
#endif
