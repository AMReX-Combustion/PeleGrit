#!/usr/bin/env python3 

import cantera as ct

gas=ct.Solution("mechanism.cti")
print("Num species   is %4d" % gas.n_species   )
print(gas.species_names)

# decane is 0.00056
# 1.0-0.00056 = 0.99944
# O2 is 0.99944*1.0/3.29 = 0.30378
# N2 is 0.69566

#gas.TPY=500.0,1.6e5,'NC10H22:0.00056,O2:0.30378,N2:0.69566'
gas.TPY=600.0,1.0e5,'NC10H22:0.0005,O2:0.23,N2:0.7695'

print("all the units are in SI")
print("T is %12.5e" % gas.T)
print("Rho is %12.5e" % gas.density)
print("D is ")
print (gas.mix_diff_coeffs)
#print (gas.mix_diff_coeffs_mass)
#print (gas.mix_diff_coeffs_mole)
print ("rhoD is ")
print(gas.density*gas.mix_diff_coeffs)
print("mu is %12.5e" % gas.viscosity)
print("mu/rho is %12.5e" % (gas.viscosity/gas.density))
print("lambda is %12.5e" % gas.thermal_conductivity)



