#!/usr/bin/env python3
# coding=UTF-8

"""
NAME
        surfrxn.py - Extract thermochemistry information 
                     from a neb calculation.

SYNTAX
        surfrxn.py OURCAR.1 OURCAR.2 OUTCAR.3 [Temperature]
        e.g. surfrxn.py IS/OURCAR TS/OURCAR FS/OUTCAR T=900K


DESCRIPTION
        Calculate pre-exponential factors, rate constants and
        thermodynamic propertie for a surface reactions at 
        a specific temperature using the harmonical approximation 
        and transition state theory.
        
        output :
            reaction.dat   
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com
    
"""

import sys
import numpy as np
import vasp_modules.assister as ast
import vasp_modules.vaspthermo as vthermo
from  vasp_modules.units import k, h, kb   
 
if __name__ == "__main__":
    #Debug	
    #sys.argv = [0, 'IS/OUTCAR', 'TS/OUTCAR', 'FS/OUTCAR']
    #T = 900
    
    if len(sys.argv) == 5:
        T = float(sys.argv[-1].strip('T').strip('K').strip('='))
    elif len(sys.argv) == 4:
        T = 298.15
        print ('No temperature is specified')
        print ('T = %.2f K will be used' %(T))
       
    else:
        print ("An exception keyword occurred")
        print ("Try syntax: surfrxn.py OURCAR.1 OURCAR.2 OUTCAR.3 [Temperature]")
        print ("e.g. surfrxn.py IS/OURCAR TS/OURCAR FS/OUTCAR T=900K")
        raise SyntaxError('Invalid syntax') 

    ini = vthermo.HarmonicThermo(sys.argv[1])  
    ts  = vthermo.HarmonicThermo(sys.argv[2]) 
    fs  = vthermo.HarmonicThermo(sys.argv[3]) 
        
    q_ini = ini._vibrational_partition_function(T)
    q_ts  = ts._vibrational_partition_function(T)
    q_fs  = fs._vibrational_partition_function(T)
    
    H_ini   = ini.get_enthalpy(T) #in eV
    H_ts    = ts.get_enthalpy(T) #in eV
    H_fs    = fs.get_enthalpy(T) #in eV
    
    S_ini   = ini.get_entropy(T) #in eV/K
    S_ts    = ts.get_entropy(T) #in eV/K
    S_fs    = fs.get_entropy(T) #in eV/K
    
    G_ini   = ini.get_gibbs_energy(T) #in eV
    G_ts    = ts.get_gibbs_energy(T) #in eV
    G_fs    = fs.get_gibbs_energy(T) #in eV   
    
    dH = H_fs - H_ini  #enthalpy change; in eV
    dS = S_fs - S_ini  #entropy change; in eV/K
    dG = G_fs - G_ini  #Gibbs energy change; in eV
    Ea = H_ts - H_ini  #activation energy; in eV 
    Ga = G_ts - G_ini  #Gibbs activation energy; in eV  
    TdS = T * dS #in eV
    
    f = k * T / h #frequency factor; in s-1
    Afwd = f * ( q_ts / q_ini )
    Arev = f * ( q_ts / q_fs)
    kfwd = Afwd * np.exp( -Ea / (kb * T) )# in s-1
    krev = Arev * np.exp( -(Ea - dH) / (kb * T) )# in s-1
    Keq = krev / kfwd

    fmt0 = '%-15s %15.6e'    
    fmt1 = '%-15s %15.6f'
    values = {'Temperature': 'T = %.2f' %(T),
              'f': fmt0 %('kT/h  =', f),
              'Q_IS': fmt0 %('q_IS =', q_ini),
              'Q_TS': fmt0 %('q_TS =', q_ts),
              'Q_FS': fmt0 %('q_FS =', q_fs),
              'Q_TS/Q_IS': fmt0 %('q_TS/q_IS =', q_ts / q_ini),
              'Q_TS/Q_FS': fmt0 %('q_TS/q_FS =', q_ts / q_fs),
              'Afwd': fmt0 %('Afwd  =', Afwd ),
              'Arev': fmt0 %('Arev  =', Arev),
              'kfwd': fmt0 %('kfwd  =', kfwd ),
              'krev': fmt0 %('krev  =', krev),
              'Keq': fmt0 %('Keq  =', Keq),  
              'Ea': fmt1 %('Ea = ', Ea),              
              'Ga': fmt1 %('Ga = ', Ga),   
              'dH': fmt1 %('dH = ', dH),
              'dS': fmt1 %('dS = ', dS),
              'TdS': fmt1 %('TdS = ', TdS),
              'dG': fmt1 %('dG = ', dG),    
          }
    
    
    thermoinfo="""
    
###############################################################################
#                                                                             #      
#   Thermodynamic properties for a surace reaction were derived               #
#   from ab-initio principles based on the harmonical approximation and       #
#   transition state theory, in whcich all degrees of freedom are treated     #
#   as the harmonic oscillator, and reaction constants are estimated using    #
#   Eyring  eqtation                                                          #
#                                                                             #
#   For more details, please see:                                             #
#       https://en.wikipedia.org/wiki/Harmonic_oscillator                     #    
#       https://en.wikipedia.org/wiki/Vibrational_partition_function          #
#       https://en.wikipedia.org/wiki/Transition_state_theory                 #
#       https://en.wikipedia.org/wiki/Eyring_equation                         # 
#                                                                             #
###############################################################################
    
==============================================================================
A surface reaction occures at %(Temperature)s K:

The fequency factor: 
%(f)s (s-1)

Vibrational partition functions:
%(Q_IS)s
%(Q_TS)s
%(Q_FS)s

Ratios of vibrational partition functions:
%(Q_TS/Q_IS)s
%(Q_TS/Q_FS)s    
    
Pre-exponential factors:  
%(Afwd)s
%(Arev)s  
 
Reaction constants:  
%(kfwd)s (s-1)
%(krev)s (s-1) 

Equilibrium constant:    
%(Keq)s
  
------------------------------------------------------------------------------
Thermodynamic properties:
%(Ea)s (eV)
%(Ga)s (eV)
%(dH)s (eV)
%(dS)s (eV/K)
%(TdS)s (eV)
%(dG)s (eV)
===============================================================================
"""
	#Write rxn.dat
    ast.print_to_file(thermoinfo %(values), 'rxn.dat', mode='w', sep='\n')
    print (thermoinfo %(values))

 

  

    
    
    
    
    