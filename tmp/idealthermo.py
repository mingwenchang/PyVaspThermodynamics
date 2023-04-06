#!/usr/bin/env python3
# coding=UTF-8

"""
Created on Thu Sep 12 09:44:40 2019
NAME
        idealthermo.py - Calculate thermodynamic properties of solid states

SYNTAX
        idealthermo.py [options] [CONTCAR/POSCAR]
        e.g. idealthermo.py
        e.g. idealthermo.py CONTCAR 
        e.g. idealthermo.py P=1.00bar T=300K spin=0 symmetry=1 geom=linear 
        e.g. idealthermo.py P=1.00bar T=300K spin=0 symmetry=1 geom=linear CONTCAR


DESCRIPTION
        Extract the thermodynamic properties from a VASP calculation. 

        output :
            zpe.dat   : contains vibrational frequencies and zero point energy 
            thermo.dat : contains U, S, H, F, G at a specific temperature
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import sys
import numpy as np
import vasp_modules.vasp_io2 as vio
import vasp_modules.assister as ast
from ase.thermochemistry import IdealGasThermo
from ase import units
from ase.io import read



def _heat_capacity(T, geom, vib_energies):
    Cv = 0 
    
    #translational heat capacity (3-d gas) 
    Cv += (3 / 2) * units.kB 
    
    #rotational heat capacity
    if geom == 'nonlinear':  
        Cv += (3 / 2) * units.kB
    elif geom == 'linear':
        Cv += units.kB
    else:# geom == 'monatomic':
        Cv += 0 #no rotational modes for monatomic 
    
    #vibrational heat capacity
    for energy in vib_energies:
        theta = energy / units.kB
        f = ( theta / T )**2 * ( np.exp( -theta / (2*T) ) / ( 1 - np.exp( -theta / T ) ) )**2 
        Cv += units.kB * f
    
    Cp = Cv + units.kB
    
    return Cv, Cp


if __name__ == "__main__":
    
    #Initialization 
    filename = 'POSCAR'
    geom = 'nonlinear' #linear or non-linear molecule 
    T = 298.15  #Unit in K
    P = 101325 #unit in pa
    sigma = 1 #symmetry number 
    spin = 0 #spin 
    
    argulen=len(sys.argv)
    if argulen > 1:
        for arg in sys.argv[1:]:
            arg = arg.lower()
            if arg.startswith('poscar') or arg.startswith('contcar'):
                filename = arg.upper()
            elif arg.startswith('temperature=') or arg.startswith('t='):
                arg = arg.strip('k').split('=')
                T = float(arg[-1]) #Unit in K
            elif arg.startswith('pressure=') or arg.startswith('p='):
                arg = arg.strip('bar').split('=')
                P = float(arg[-1])*101325 #unit in pa
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
                print ('Try syntax: idealthermo.py [options] [CONTCAR/POSCAR]')
                print ("e.g. idealthermo.py CONTCAR")
                print ("e.g. idealthermo.py P=1.00bar T=300K spin=0 symmetry=1 geom=linear CONTCAR")
                raise SyntaxError('invalid syntax') 
    
    print ('\n\n\n') 
    print ('Following parameters will be adopted:\n'+
           'filename: %s\n' %(filename) +
           'temperture: %s K\n' %(T)+ 
           'pressure: %s pa\n'%(P) + 
           'spin: %s\n' %(spin)+ 
           'symmetry: %s\n' %(sigma)+
           'geom: %s\n' %(geom))
    
    atoms =read(filename)
    etotal=  vio.get_energy(mode='all')[0]
    vibframe = vio.extra_vibinfo()
    if geom == 'linear' :
        vibfreqs = [0.001 * vib for vib in vio.get_freqs(vibframe, 'meV')][:-5]
        vibinfo  = [line for line in vibframe if 'f' in line][:-5]
    elif geom == 'nonlinear':
        vibfreqs = [0.001 * vib for vib in vio.get_freqs(vibframe, 'meV')][:-6]
        vibinfo  = [line for line in vibframe if 'f' in line][:-6]
    else:
         vibfreqs = [0.000]
         vibinfo = ['No vibrational modes']

         
      
    idealthermo = IdealGasThermo(vib_energies=vibfreqs,
                                 potentialenergy=etotal,
                                 atoms=atoms,
                                 geometry=geom,
                                 symmetrynumber=sigma,
                                 spin=spin)
    
    ZPE = idealthermo.get_ZPE_correction()
    H = idealthermo.get_enthalpy(temperature=T, verbose=False)  
    S = idealthermo.get_entropy(temperature=T, pressure=P, verbose=False)
    G = idealthermo.get_gibbs_energy(temperature=T, pressure=P, verbose=False)
    
    TS = T*S
    Cp = _heat_capacity(T, geom, vibfreqs)[-1]
    U = H - units.kB * T #U = H - PV = H - nRT
    F = U - T*S
   
    
    #Write zpe.dat
    vibinfo = vibinfo + ['Zero Point Energy: %.2f eV' %(ZPE)]
    ast.print_to_file(vibinfo, 'zpe.dat', mode='w', sep='\n')
    
    #Write thermo.dat 
    fmt0 = '%-15s %15.6f eV'
    fmt1 = '%-15s %15.2f eV'
    fmt2 = '%-15s %15.6f eV/K'
    values = {'Temperature': 'T = %.2f' %(T),
              'Pressure': 'P = %.2f'%(P),
              'E_total': fmt0 %('E_total:', etotal),
              'E_ZPE': fmt1 %('E_ZPE:', ZPE),
              'T*S': fmt1 %('TS:', TS ),
              'U': fmt0 %('U(T) = ', U),
              'S': fmt2 %('S(T) = ', S),
              'Cp': fmt2 %('Cp(T) = ', Cp),
              'H': fmt0 %('H(T) = ', H),
              'F': fmt0 %('F(T) = ', F),
              'G': fmt0 %('G(T) = ', G)}
           
    thermoinfo="""
###############################################################################
#                                                                             # 
#   Ccalculating thermodynamic properties of a molecule                       #
#   based on statistical mechanical treatments in the ideal gas               #
#   approximation.                                                            # 
#                                                                             #
#   Definition of thermodynamic properties:                                   #
#                                                                             # 
#           H = U + PV  = U + nRT                                             #
#                                                                             #
#           F = U - TS                                                        # 
#                                                                             #
#           G = U + PV - TS                                                   #
#                                                                             #
#                                                                             # 
###############################################################################


==============================================================================
Thermodynamic properties of an ideal-gas molecule
at %(Temperature)s K and %(Pressure)s Pa:

%(E_total)s
%(E_ZPE)s
%(T*S)s
    
------------------------------------------------------------------------------

%(Cp)s
%(S)s
%(U)s
%(H)s
%(F)s
%(G)s
==============================================================================
"""

    ast.print_to_file(thermoinfo %(values), 'thermo.dat', mode='w', sep='\n')
    
    #Print vibinfo and thermoinfo
    print ('\n\n\n')   
    print ('Vibrational frequencies:')
    for line in vibinfo:
        print (line)
        
    print ('\n\n\n')   
    print ('Thermochemistry properties:')    
    print (thermoinfo %(values))
