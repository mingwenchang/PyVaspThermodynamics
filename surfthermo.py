#!/usr/bin/env python3
# coding=UTF-8

"""
NAME
        surfthermo.py - Calculate thermodynamic properties of an adsorbate

SYNTAX
        surfthermo.py [OUTCAR] [Temperature] 
        e.g. surfthermo.py 600K
        e.g. surfthermo.py OUTCAR T=600K


DESCRIPTION
        Extract thermodynamic properties from a VASP calculation. 
        
        output :
            zpe.dat: vibrational frequencies and zero point energy 
            thermo.dat: partition functions, U, S, H, F, G etc. at a specific temperature
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import sys
import vasp_modules.assister as ast
import vasp_modules.vaspthermo as vthermo
 
if __name__ == "__main__":
    
    T = 298.15
    outcar ='OUTCAR'
    argulen=len(sys.argv)
    if argulen > 1:
        for arg in sys.argv[1:]:
            path = arg
            arg = arg.lower()
            if 'outcar' in arg:
                outcar = path
            elif arg.startswith('temperature=') or arg.startswith('t='):
                arg = arg.strip('k').split('=')
                T = float(arg[-1]) #Unit in K
            else:
                print ("An exception keyword occurred")
                print ('Try syntax: surfthermo.py  [OUTCAR] [Temperature]')
                print ("e.g. surfthermo.py T=900K")
                print ("e.g. surfthermo.py OUTCAR")
                print ("e.g. surfthermo.py OUTCAR T=900K")
                raise SyntaxError('invalid syntax') 
    
    solid = vthermo.HarmonicThermo(outcar)     
    q_vib =  solid._vibrational_partition_function(T) #Vibtational
    chemform = solid.atoms.get_chemical_formula()
    
    etotal = solid.etotal #in eV
    ZPE = solid.get_ZPE_correction() #in eV
    U   = solid.get_internal_energy(T) #in eV
    H   = solid.get_enthalpy(T) #in eV
    S   = solid.get_entropy(T) #in eV/K
    G   = solid.get_gibbs_energy(T) #in eV
    F   = solid.get_helmholtz_energy(T) #in eV
    
    TS = T*S #in eV
    PV = 0.00
    Cv = solid.get_heat_capacity(T) #in eV/K            

    fmt0 = '%-15s %15.6e'    
    fmt1 = '%-15s %15.6f'
    values = {'molecule': 'Molecule = %s' %(chemform),
              'T': 'Temperature = %.2f' %(T),
              'Q_vib': fmt0 %('q_vib =', q_vib),
              'E_total': fmt1 %('E_total = ', etotal),
              'E_ZPE': fmt1 %('E_ZPE = ', ZPE),
              'T*S': fmt1 %('TS = ', TS ),
              'Cv': fmt1 %('Cv(T) = ', Cv),
              'U': fmt1 %('U(T) = ', U),
              'S': fmt1 %('S(T) = ', S),
              'H': fmt1 %('H(T) = ', H),
              'F': fmt1 %('F(T) = ', F),
              'G': fmt1 %('G(T) = ', G)}   


    
    thermoinfo="""
###############################################################################
#                                                                             #      
#   Thermodynamic properties of an adsorbate were derived from                # 
#   ab-initio principles based on the harmonical approximation,               #
#   in whcich all degrees of freedom are treated as the harmonic              #
#   oscillator                                                                #
#                                                                             #
#   For more details, please see:                                             #
#       https://en.wikipedia.org/wiki/Harmonic_oscillator                     #  
#       https://en.wikipedia.org/wiki/Vibrational_partition_function          #
#                                                                             #
###############################################################################

==============================================================================
System: 
%(molecule)s

Enviroment:  
%(T)s (K)

------------------------------------------------------------------------------
Partition function: 
%(Q_vib)s

------------------------------------------------------------------------------
Thermodynamic properties:
%(E_total)s (eV)
%(E_ZPE)s (eV)
%(Cv)s (eV/K)
%(T*S)s (eV)
%(S)s (eV/K)
%(U)s (eV)
%(H)s (eV)
%(F)s (eV)
%(G)s (eV)
===============================================================================
"""

	#Write thermo.dat
    ast.print_to_file(thermoinfo %(values), 'thermo.dat', mode='w', sep='\n')
    print (thermoinfo %(values))

    #Write zpe.dat
    vibinfo = solid.vibinfo + ['Zero Point Energy: %.2f eV' %(ZPE)]
    ast.print_to_file(vibinfo, 'zpe.dat', mode='w', sep='\n')





