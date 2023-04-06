#!/usr/bin/env python3
# coding=UTF-8

"""
NAME
        idealthermo.py - Calculate molecular Partition functions 
                         for an ideal gas molecule

SYNTAX
        idealthermo.py [options]
        e.g. idealthermo.py
        e.g. idealthermo.py OUTCAR 
        e.g. idealthermo.py OUTCAR P=1.00bar A=1.00A^2 T=300K spin=0 symmetry=1 geom=linear 
        e.g. idealthermo.py P=1.00bar A=1.00A^2 T=300K spin=0 symmetry=1 geom=linear 

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
import numpy as np
import vasp_modules.assister as ast
import vasp_modules.vaspthermo as vthermo
from  vasp_modules.units import pi, k, mu, kb    
 


if __name__ == "__main__":
     #Initialization 
    outcar = 'OUTCAR'
    geom = 'nonlinear' #linear or non-linear molecule 
    T = 298.15  #Unit in K
    P = 101325 #unit in pa
    sigma = 1 #symmetry number 
    spin = 0 #spin 
    A = 1.00E-20 # Area 
    
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
            elif arg.startswith('pressure=') or arg.startswith('p='):
                arg = arg.strip('bar').split('=')
                P = float(arg[-1])*101325 #unit in pa
            elif arg.startswith('area=') or arg.startswith('a='):
                arg = arg.strip('A^2').split('=')
                A = float(arg[-1]) * 1E-20 #unit in m^2
            elif arg.startswith('symmetry=') or arg.startswith('sigma='):
                arg = arg.split('=')
                sigma = int(arg[-1])
            elif arg.startswith('geometry=') or arg.startswith('geom='):
                arg = arg.split('=')
                geom = arg[-1] 
            elif arg.startswith('spin='):
                arg = arg.split('=')
                spin = int(arg[-1])
            else:
                print ("An exception keyword occurred")
                print ('Try syntax: idealthermo.py [options]')
                print ("e.g. idealthermo.py")
                print ("e.g. idealthermo.py P=1.00bar T=300K A=0.10nm^2 spin=0 symmetry=1 geom=linear")
                print ("e.g. idealthermo.py OUTCAR P=1.00bar T=300K A=0.10nm^2 spin=0 symmetry=1 geom=linear")
                raise SyntaxError('invalid syntax') 

    gas = vthermo.IdealGasThermo(outcar, geom, sigma, spin)
    chemform = gas.atoms.get_chemical_formula()
    m = gas.mass 
    
    kads =  (P * A) /np.sqrt(2 * pi * m * k * T) #Hertz-Knudesn equation 
    q_t2D = gas._2D_translational_partition_function(T, A) #2D-translational
    q_t3D = gas._3D_translational_partition_function(T, P) #3D-translational
    q_rot = gas._rotational_partition_function(T) #Rotational
    q_vib = gas._vibrational_partition_function(T) #Vibtational
    q_ele = gas._electronic_partition_function() #Electronic
    qtotal_2D = q_t2D *  q_rot * q_vib * q_ele
    qtotal_3D = q_t3D *  q_rot * q_vib * q_ele
    
    etotal = gas.etotal
    ZPE = gas.get_ZPE_correction() #in eV
    U   = gas.get_internal_energy(T) #in eV
    H   = gas.get_enthalpy(T) #in eV
    S   = gas.get_entropy(T, P) #in eV/K
    G   = gas.get_gibbs_energy(T, P) #in eV
    F   = gas.get_helmholtz_energy(T, P) #in eV
    
    TS = T*S #in eV
    Cv = gas.get_heat_capacity(T) #in eV/K
    Cp = Cv + kb #in eV/K
    
    fmt0 = '%-15s %15.6e'    
    fmt1 = '%-15s %15.6f'
    fmt2 = '%-15s %15.6f'
    
    values = {'molecule': 'Molecule = %s' %(chemform),
              'geom' : 'Geometry = %s' %(geom),
              'sigma': 'Symmetry num. = %s' %(sigma),
              'spin': 'Spin = %s' %(spin),
              'M': 'Mass = %.2f' %( m / mu),
              'T': 'Temperature = %.2f' %(T),
              'P': 'Pressure = %.2f' %(P),
              'A': fmt0  %('Effective Area: ', A), 
              'kads': fmt0 %('kads = ', kads),
              'Q_t3D': fmt0 %('q_trans3D =', q_t3D),
              'Q_t2D': fmt0 %('q_trans2D =', q_t2D),
              'Q_rot': fmt0 %('q_rot =', q_rot ),
              'Q_vib': fmt0 %('q_vib =', q_vib),
              'Q_ele': fmt0 %('q_ele =', q_ele),
              'Qtotal_2D': fmt0 %('qtotal_2D =', qtotal_2D),
              'Qtotal_3D': fmt0 %('qtotal_3D =', qtotal_3D),
              'E_total': fmt1 %('E_total = ', etotal),
              'E_ZPE': fmt1 %('E_ZPE = ', ZPE),
              'T*S': fmt1 %('TS = ', TS ),
              'Cp': fmt1 %('Cp(T) = ', Cp),
              'U': fmt1 %('U(T) = ', U),
              'S': fmt1 %('S(T) = ', S),
              'H': fmt1 %('H(T) = ', H),
              'F': fmt1 %('F(T) = ', F),
              'G': fmt1 %('G(T) = ', G)}              
              
    thermoinfo="""
###############################################################################
#                                                                             #      
#   Thermodynamic properties of an ideal-gas were derived from                #
#   ab-initio principles using stastistical thermodynamic partition           #
#   function.                                                                 #
#                                                                             #
#   For more details, please see:                                             #
#       https://en.wikipedia.org/wiki/Translational_partition_function        #
#       https://en.wikipedia.org/wiki/Rotational_partition_function           #
#       https://en.wikipedia.org/wiki/Vibrational_partition_function          #
#                                                                             #
###############################################################################

==============================================================================
System: 
%(molecule)s
%(M)s (amu)
%(geom)s
%(sigma)s
%(spin)s

Enviroment:  
%(T)s (K)
%(P)s (Pa)
%(A)s (m^2)
------------------------------------------------------------------------------
Partition functions for each component:
%(Q_t2D)s 
%(Q_t3D)s 
%(Q_rot)s
%(Q_vib)s
%(Q_ele)s

Total artition function:
%(Qtotal_2D)s
%(Qtotal_3D)s

The rate constant: 
%(kads)s (for the non-activated adsorption)
------------------------------------------------------------------------------
Thermodynamic properties:
%(E_total)s (eV)
%(E_ZPE)s (eV)
%(Cp)s (eV/K)
%(T*S)s (eV)
%(S)s (eV/K)
%(U)s (eV)
%(H)s (eV)
%(F)s (eV)
%(G)s (eV)
==============================================================================
"""
    #Write thermo.dat
    ast.print_to_file(thermoinfo %(values), 'thermo.dat', mode='w', sep='\n')
    print (thermoinfo %(values))
	
    #Write zpe.dat
    vibinfo = gas.vibinfo + ['Zero Point Energy: %.2f eV' %(ZPE)]
    ast.print_to_file(vibinfo, 'zpe.dat', mode='w', sep='\n')	