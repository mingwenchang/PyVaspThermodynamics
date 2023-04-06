#!/usr/bin/env python3
# coding=UTF-8

"""
Created on Thu Sep 12 09:44:40 2019
NAME
        solidthermo.py - Calculate thermodynamic properties of solid states

SYNTAX
        solidthermo.py [Temperature]
        e.g. vthermo.py 600K


DESCRIPTION
        Extract the total DOS and projected DOS from a VASP calculation. All DOS are summed on
        spin up and down.

        output :
            zpe.dat   : contains vibrational frequencies and zero point energy 
            thermo.dat : contains U, S, H, F, G at a specific temperature
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import sys
import vasp_modules.vasp_io2 as vio
import vasp_modules.assister as ast
from ase.thermochemistry import HarmonicThermo

if __name__ == "__main__":
    
    argulen=len(sys.argv)
    if argulen == 1:
        T = 298.15
        print ('No temperature was specified, T = 298.15 K is adpoted!!')
    else:
        arg =sys.argv[1].lower() 
        try:
            T = float(arg.strip('k').strip('temperature=').strip('t='))
        except:
            print("An exception keyword occurred")
            print ('Try syntax: solidthermo.py Temperature')
            print ('e.g.: solidthermo.py T=600K')
            raise SyntaxError('invalid syntax')       
      
    etotal=  vio.get_energy(mode='a')[0]
    vibframe = vio.extra_vibinfo()
    vibfreqs = [0.001 * vib for vib in vio.get_freqs(vibframe, 'meV') if vib > 0]  
     
    harmonics = HarmonicThermo(vib_energies=vibfreqs, potentialenergy=etotal)
    ZPE = harmonics.get_ZPE_correction()
    dU_vib = harmonics._vibrational_energy_contribution(temperature=T)
    U = harmonics.get_internal_energy(temperature=T, verbose=False) # U(0) + ZPE + dU_v
    S = harmonics.get_entropy(temperature=T, verbose=False) #vibrational entropy contribution
    TS = T*S
    PV = 0.00
    
    F = U - TS
    G = F + PV
    H = U + PV
    
    #Write zpe.dat
    vibinfo = [line for line in vibframe if 'f' in line] + ['Zero Point Energy: %.2f eV' %(ZPE)]
    ast.print_to_file(vibinfo, 'zpe.dat', mode='w', sep='\n')
    
    
    #Write thermo.dat
    fmt0 = '%-15s %15.6f eV'
    fmt1 = '%-15s %15.2f eV'
    fmt2 = '%-15s %15.6f eV/K'
    values = {'Temperature': 'T = %.2f' %(T),
              'E_total': fmt0 %('E_total:', etotal),
              'E_ZPE': fmt1 %('E_ZPE:', ZPE),
              'E_vib': fmt1 %('E_vib(0->T):', dU_vib ),
              'T*S': fmt1 %('TS:', TS ),
              'P*V': fmt1 %('PV (neglected):', PV ),
              'U': fmt0 %('U(T) = ', U),
              'S': fmt2 %('S(T) = ', S),
              'H': fmt0 %('H(T) = ', H),
              'F': fmt0 %('F(T) = ', F),
              'G': fmt0 %('G(T) = ', G)}
    
    
    thermoinfo="""
###############################################################################
#                                                                             # 
#   In the harmonic limit, all degrees of freedom are treated harmonically.   #
#   For solid states, the pV term usually small, which is neglectable         # 
#   In this case, Enthalpy(H) and Gibbs free energy (G) can be interpreted    #
#   as internal energy (U) and Helmholtz free energy (H), respectively.       # 
#                                                                             #
#    proof:                                                                   #
#       The definition of Enthalpy:                                           # 
#           H = U + PV                                                        #
#                                                                             #
#       The definition of Helmholtz free energy:                              # 
#           F = U - TS                                                        # 
#                                                                             #
#       The definition of Gibbs free energy:                                  # 
#           G = U + PV - TS  = F + PV                                         #
#                                                                             #
#       Because PV is usually small for solids,                               #
#       => G = F + PV = F                                                     #
#       => H = U + PV = U                                                     #
#                                                                             # 
###############################################################################


==============================================================================
Thermodynamic properties of solid states
based on the harmonic limit at %(Temperature)s K:

%(E_total)s
%(E_ZPE)s
%(E_vib)s
%(T*S)s
%(P*V)s  
    
------------------------------------------------------------------------------

%(S)s
%(U)s
%(H)s
%(F)s
%(G)s
==============================================================================
"""
    ast.print_to_file(thermoinfo %(values), 'thermo.dat', mode='w', sep='\n')
    
    #Print vibinfo and thermoinfo
    print ('Vibrational frequencies:')
    for line in vibinfo:
        print (line)
    
    print ('\n\n\n')   
    print ('Thermochemistry properties:')    
    print (thermoinfo %(values))









