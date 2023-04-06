#!/usr/bin/env python3
# coding=UTF-8

"""
Created on Thu Jan 29 09:44:40 2020
NAME
        solidprefactor.py - 

SYNTAX
        solidprefactor.py OURCAR.IS [OURCAR.TS] [OUTCAR.FS] [Temperature] 
        e.g. solidprefactor.py OURCAR.IS OURCAR.TS OURCAR.FS T=300


DESCRIPTION
        Calculate pre-exponential factors for surface reactions 
        according to vibrational partition function in
        statistical thermodynamics

        output :
            prefactor.dat   
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com
    
"""

import sys, os
import numpy as np
import vasp_modules.vasp_io2 as vio
import vasp_modules.assister as ast

k = 8.617333264E-005 #unit in eV⋅K−1 
h = 4.135667696E-015 #unit in eV⋅S
rootdir = os.getcwd()


def Qvib(vmodes, T=300): 
    Q = 1
    q = lambda v, T:  1 / (1-np.exp(-v/(k*T))) #no-zep correction 
    #q = lambda v, T:  np.exp(-v/(2*k*T)) / (1-np.exp(-v/(k*T))) ##with zep correction 
    for v in vmodes:
            Q = Q*q(v, T)
    return Q

#arguments = ['OUTCAR.IS', 'OUTCAR.TS', 'OUTCAR.FS', 'T=900']
#arguments = ['OUTCAR.IS', 'OUTCAR.TS']
arguments = sys.argv  
arg = arguments[-1].lower() 
try:
    T = float(arg.strip('k').strip('temperature=').strip('t='))
    outcars = arguments[1:-1]
except:
    T = 298.15
    outcars = arguments[1:]
    
vibfuns = [ ]
for outcar in outcars:
    #outcar = '%s/%s' %(rootdir, outcar)
    vibframe = vio.extra_vibinfo(outcar) 
    vibfreqs = [0.001 * vib for vib in vio.get_freqs(vibframe, 'meV') if vib > 0] #unit in eV  
    Q = Qvib(vibfreqs, T) 
    vibfuns.append(Q) 

fmt0 = '%-15s %15.6e'    
if len(vibfuns) == 3:
    Q_IS = vibfuns[0] 
    Q_TS = vibfuns[1]
    Q_FS = vibfuns[2] 
    
    f = k * T / h #frequency factor in S
      
    values = {'Temperature': 'T = %.2f' %(T),
              'f': fmt0 %('kT/h  =', f),
              'Q_IS': fmt0 %('Q_IS =', Q_IS),
              'Q_TS': fmt0 %('Q_TS =', Q_TS),
              'Q_FS': fmt0 %('Q_FS =', Q_FS),
              'Q_TS/Q_IS': fmt0 %('Q_TS/Q_IS =', Q_TS / Q_IS),
              'Q_TS/Q_FS': fmt0 %('Q_TS/Q_FS =', Q_TS / Q_FS),
              'A_fwd': fmt0 %('A_fwd  =', f * ( Q_TS / Q_IS ) ),
              'A_rev': fmt0 %('A_rev  =', f * ( Q_TS / Q_FS ) ),
          }
    
    
    thermoinfo="""
###############################################################################
#                                                                             #      
#   Vibrational partition functions derived from frequency anlysis            #
#   were used to calculate the pre-exponential factorsfor a                   #
#   surface reactoion                                                         # 
#                                                                             #
#   For more details, please see:                                             #
#       https://en.wikipedia.org/wiki/Vibrational_partition_function          #
#                                                                             #
###############################################################################

------------------------------------------------------------------------------

The fequency factor at %(Temperature)s K:
%(f)s

Vibrational partition functions at %(Temperature)s K:
%(Q_IS)s
%(Q_TS)s
%(Q_FS)s

Ratios of vibrational partition functions at %(Temperature)s K:
%(Q_TS/Q_IS)s
%(Q_TS/Q_FS)s    
    
Pre-exponential factors at %(Temperature)s K:    
%(A_fwd)s
%(A_rev)s  
   
------------------------------------------------------------------------------
"""
    ast.print_to_file(thermoinfo %(values), 'prefactor.dat', mode='w', sep='\n')
    print (thermoinfo %(values))
    
else:
    print ('Vibrational partition functions at %.2f K:' %(T))
    ast.print_to_file('Vibrational partition functions at %.2f K:' %(T), 'prefactor.dat', mode='w', sep='\n')
    for i, vibfun in enumerate(vibfuns):
        i = i + 1 
        print (fmt0 %('No.'+str(i)+'-OUTCAR:', vibfun))
        ast.print_to_file(fmt0 %('No.'+str(i)+'-OUTCAR:', vibfun), 'prefactor.dat', mode='a', sep='\n')
 

  

    
    
    
    
    