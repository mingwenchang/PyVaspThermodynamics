#!/usr/bin/env python3
# coding=UTF-8

"""
Created on Thu Sep 12 09:44:40 2019
NAME
        idealpartfuns.py - Calculate molecular Partition functions 
                           for an ideal gas molecule

SYNTAX
        idealpartfuns.py [options] [CONTCAR/POSCAR]
        e.g. idealpartfuns.py
        e.g. idealpartfuns.py CONTCAR 
        e.g. idealpartfuns.py P=1.00bar A=1.00A^2 T=300K spin=0 symmetry=1 geom=linear 
        e.g. idealpartfuns.py P=1.00bar A=1.00A^2 T=300K spin=0 symmetry=1 geom=linear CONTCAR


DESCRIPTION
        Extract the thermodynamic properties from a VASP calculation. 
        
        output :
            idealpartfuns.dat : contains values of 2D-translational, 
            3D-translational rotational and vibrational partitions, etc.
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import sys
import numpy as np
import vasp_modules.vasp_io2 as vio
import vasp_modules.assister as ast
from scipy.constants import c, k, h, pi, m_u

def Qtrans_3D(m, T=300, P=1.00E+05):
    V = ( k * T)/P
    L = h / ( 2 * pi * m * k * T)**0.5 #Thermal de Broglie wavelength
    Q = V / L**3
    return Q

def Qtrans_2D(m, A=1.00E-20, T=300):
    Q = A * ( 2 * pi * m * k * T) / (h**2)
    return Q

rc = lambda I:  h / (8* pi**2 * c * I) #inertia of momentum to the rotational constant
def Qrot(inertias, sg=1, T=300, geom= 'nonlinear'):
    if geom == 'linear' : #linear molecule
        Ia = inertias
        Q = (8 * pi**2 * Ia *k * T) / (sg * h**2)
    elif  geom == 'nonlinear':
        Ia, Ib, Ic = inertias
        Q = (1/sg) * (pi * Ia * Ib * Ic)**0.5 * ((8 * pi**2 * k * T) / (h**2))**(3/2)
    else:
        Q = 1
    return Q

def Qvib(vmodes, T=300): 
    Q = 1
    q = lambda v, T:  1 / (1-np.exp(-v/(k*T))) #no-zep correction 
    #q = lambda v, T:  np.exp(-v/(2*k*T)) / (1-np.exp(-v/(k*T))) ##with zep correction 
    for v in vmodes:
        Q = Q*q(v, T)
    return Q

def Qele(s):
    Q = 2*s + 1 
    return Q

if __name__ == "__main__":
     #Initialization 
    poscar = 'POSCAR'
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
            arg = arg.lower()
            if arg.startswith('poscar') or arg.startswith('contcar'):
                filename = arg.upper()
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
                print ('Try syntax: idealthermo.py [options] [CONTCAR/POSCAR]')
                print ("e.g. idealpartfuns.py CONTCAR")
                print ("e.g. idealpartfuns.py P=1.00bar T=300K A=0.10nm^2 spin=0 symmetry=1 geom=linear CONTCAR")
                raise SyntaxError('invalid syntax') 
    
    print ('\n\n\n') 
    print ('Following parameters will be adopted:\n'+
           'files: %s and %s\n' %(poscar, outcar) +
           'temperture: %s K\n' %(T)+ 
           'pressure: %s pa\n'%(P) + 
           'effective area: %s A^2\n'%(A) + 
           'spin: %s\n' %(spin)+ 
           'symmetry: %s\n' %(sigma)+
           'geom: %s\n' %(geom))   


    atoms =vio.read_poscar(poscar)
    chemform = atoms.get_chemical_formula() #chemical formula
    vibframe = vio.extra_vibinfo(outcar) 
    vibfreqs = [0.001 * 1.60218E-19 * vib for vib in vio.get_freqs(vibframe, 'meV')]
    inertias = [(m_u / (1.00E+10)**2) * inertia for inertia in atoms.get_moments_of_inertia()]
    m = m_u * atoms.get_molecular_mass() 
    
    if geom == 'linear' :
        vibfreqs = vibfreqs[:-5]  
        inertias = max(inertias)
    elif geom == 'nonlinear':
        vibfreqs = vibfreqs[:-6] 
    else:
         vibfreqs = [0.000]
         inertias = 0
    
    f =  (P * A) /np.sqrt(2 * pi * m * k * T) #frequency factor in S
    q_t2D = Qtrans_2D(m, A, T) #2D-translational function
    q_t3D = Qtrans_3D(m, T, P) #3D-translational function
    q_rot = Qrot(inertias, sigma, T, geom) #Rotational function
    q_vib = Qvib(vibfreqs, T) #Vibrational function
    q_ele = Qele(spin)
    
    fmt0 = '%-15s %15.6e'    
    values = {'molecule': 'Molecule =  %s' %(chemform),
              'T': 'Temperature = %.2f' %(T),
              'P': 'Pressure = %.2f' %(P),
              'M': 'Molecular Mass = %.2f' %(m/m_u),
              'A': fmt0  %('Effective Area = ', A), 
              'f': fmt0 %('(P * A) /sqrt(2*pi*m*k*T) = ', f),
              'Q_t3D': fmt0 %('Q_t3D =', q_t3D),
              'Q_t2D': fmt0 %('Q_t2D=', q_t2D),
              'Q_rot': fmt0 %('Q_rot =', q_rot ),
              'Q_vib': fmt0 %('Q_vib =', q_vib),
              'Q_ele': fmt0 %('Q_ele =', q_ele)
              }
    
    
    thermoinfo="""
###############################################################################
#                                                                             #      
#   Molecular Partition functions for an ideal gas molecule were calculated   #
#   accroding to the statistical mechanics expressions.                       #
#                                                                             #
#   For more details, please see:                                             #
#       https://en.wikipedia.org/wiki/Translational_partition_function        #
#       https://en.wikipedia.org/wiki/Rotational_partition_function           #
#       https://en.wikipedia.org/wiki/Vibrational_partition_function          #
#                                                                             #
###############################################################################

------------------------------------------------------------------------------
%(molecule)s
%(M)s amu
%(T)s K
%(P)s Pa
%(A)s m^2

------------------------------------------------------------------------------
The fequency factor:
%(f)s

The 2D translational partition function: 
%(Q_t2D)s

The 3D translational partition function: 
%(Q_t3D)s

The rotational partition function: 
%(Q_rot)s

The vibrational partition function: 
%(Q_vib)s

The electronic partition function: 
%(Q_ele)s

------------------------------------------------------------------------------
"""

    ast.print_to_file(thermoinfo %(values), 'idealpartfuns.dat', mode='w', sep='\n')
    print (thermoinfo %(values))