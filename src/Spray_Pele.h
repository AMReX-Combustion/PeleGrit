/*---------------------------------------------------------------------------*\
  o  o o  o         |
   o  o o 	G       | Grit: The Toolbox for Lagrangian Particles
  o oo 		  R       |	with GPU Acceleration
   oo    	  I       | 
   o     	  T   	  |
-------------------------------------------------------------------------------

Description	
	Basic class for particles
\*---------------------------------------------------------------------------*/
#ifndef SPRAY_PELE_H
#define SPRAY_PELE_H

#include "Dust_Pele.h"

class Spray_Pele : public Dust_Pele {
  public:
    ScalarPointType temp;

    // constructor
    Spray_Pele() {
      temp = ScalarPointType ("temp");
    };
};
#endif
