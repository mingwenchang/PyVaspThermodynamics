#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        vaspthermo.py - Extract thermochemistry information from OUTCAR.
                           
 
DESCRIPTION
   
        This module provides common methods used in thermochemistry
        calculations. Most of codes in the module are mainly from 
        ase.thermochemistry module, but the code structure was 
        modified with specialized new features for vasp.

              
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com
    
    ASE-developers
    https://wiki.fysik.dtu.dk/ase/about.html

"""


import numpy as np
import vasp_modules.vasp_io2 as vio
from vasp_modules.units import pi, k, h, mu, kb, ang, eVtoJ    
 
class ThermoChem: #Basic class    
    def _thermal_de_Broglie_wavelength(self, T):
        L = h / np.sqrt (2 * pi * self.mass * k * T)
        return L #unit in m
    
    def _3D_translational_partition_function(self, T, P):
        V = ( k * T) / P
        L = self._thermal_de_Broglie_wavelength(T)
        Q = V / L**3
        return Q

    def _2D_translational_partition_function(self, T, A):
        Q = A * (2 * pi * self.mass * k * T) / h**2
        return Q
    
    def _rotational_partition_function(self, T):
        if self.geom == 'linear' : #linear molecule
            Ia = self.inertias
            Q = (8 * pi**2 * Ia *k * T) / (self.sigma * h**2)
        elif  self.geom == 'nonlinear':
            Ia, Ib, Ic = self.inertias
            Q = (1 / self.sigma) * (pi * Ia * Ib * Ic)**0.5 * ((8 * pi**2 * k * T) / (h**2))**(3/2)
        else:
            Q = 1
        return Q
    
    def _vibrational_partition_function(self, T): 
        Q = 1.0
        #q = lambda v, T:  1 / (1-np.exp( -v /(k * T) )) #no-zep correction 
        #q = lambda v, T:  np.exp(-v/(2*k*T)) / (1-np.exp(-v/(k*T))) #include zep correction 
        for v in self.vib_energies:
            v *= eVtoJ
            Q *= 1.0 / (1.0-np.exp( -v /(k * T) ))
        return Q
    
    def _electronic_partition_function(self):
        Q = 2 * self.spin + 1 
        return Q    
    
    def get_frequency_factor(self, T):
        f = k * T / h
        return f
        
    def get_ZPE_correction(self): 
        """Returns the zero-point correction energy in eV."""
        zpe = 0.
        for energy in self.vib_energies:
            zpe += 0.5 * energy
        return zpe 

    def _vibrational_energy_contribution(self, T): #Codes from ASE
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin. Returns the energy change
        in eV."""
        
        kbT = kb * T
        dU = 0.
        for energy in self.vib_energies:
            dU += energy / (np.exp(energy / kbT) - 1.)
        return dU

    def _vibrational_entropy_contribution(self, T): #Codes from ASE
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        S_v = 0.
        for energy in self.vib_energies:
            x = energy / (kb * T)
            S_v += x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x))
        S_v *= kb
        return S_v
    
    def _vibrational_heat_capacity_contribution(self, T):
        """Calculates the heat_capacity due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the Cv"""
        Cv = 0 
        #vibrational heat capacity
        for energy in self.vib_energies:
            theta = energy / kb  
            #f = ( theta / T )**2 * ( np.exp( -theta / (2*T) ) / ( 1 - np.exp( -theta / T ) ) )**2 
            f = ( theta / T )**2 * ( np.exp( -theta / (2*T) ) / ( 1 - np.exp( -theta / T ) ) )**2 
            Cv +=  kb * f
        return Cv
        

class IdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.
    """
    #Codes from ASE

    def __init__(self, outcar='OUTCAR', geom=None, sigma=None, spin=None):
        
        self.atoms  = vio.get_structures(outcar, mode='all')[0] #initial structure
        self.etotal = vio.get_energy(outcar, mode='all')[0] #first 
        self.geom = geom
        self.sigma = sigma
        self.spin = spin
        self.vibinfo  = [line for line in vio.extra_vibinfo(outcar)if 'f' in line]
        self.mass = self.atoms.get_molecular_mass() * mu
        
        vib_energies = [0.001 * vib for vib in vio.get_freqs(self.vibinfo, 'meV')]
        inertias = self.atoms.get_moments_of_inertia() * (mu * ang**2) 
        ntotal = sum(self.atoms.natoms)

        if geom == 'nonlinear':
            self.vib_energies = vib_energies[0: 3*ntotal-6]
            self.inertias = inertias
        elif geom == 'linear':
            self.vib_energies = vib_energies[0: 3*ntotal-5]
            self.inertias = np.max(inertias)    
        else:# geom == 'monatomic':
            self.vib_energies = [0.00]
            self.inertias = 0

    def get_internal_energy(self, temperature):
        """Returns U(T), in eV, in the ideal gas approximation
        at a specified temperature (K)."""
        
        #total energy at T = 0K
        U_0 = self.etotal

        #ZPE correction to the total energy 
        zpe = self.get_ZPE_correction()
        
        #Mean translational enery(3-d gas)
        U_t = (3. / 2.) * kb  * temperature

        #Mean rotational heat capacity
        if self.geom == 'nonlinear':  
            U_r = (3. / 2. )* kb * temperature
        elif self.geom == 'linear':
            U_r = kb * temperature
        else: # self.geom == 'monatomic':
            U_r = 0. 
        
        # vibrational heat capacity
        U_v = self._vibrational_energy_contribution(temperature)
        
        U = ( U_0 + zpe + U_t + U_r + U_v) 
        return U
    
    def get_enthalpy(self, temperature):
        """Returns H(T), in eV, in the ideal gas approximation
        at a specified temperature (K)."""
        
        U = self.get_internal_energy(temperature)
        H = U + kb * temperature  # H = U + PV = U + nRT 
        return H

    def get_entropy(self, temperature, pressure):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""
        #Codes from ASE 
        if self.sigma is None or self.spin is None:
            raise RuntimeError('symmetrynumber and spin must be '
                               'specified for entropy and free energy '
                               'calculations.')
        
        #Translation contributes to entropy (Sackur-Tetrode equation):
        #S(T) = kln(Qtrans) + 5/2
         # unit in kg
        qtrans = self._3D_translational_partition_function(temperature, pressure) 
        S_t = kb * np.log(qtrans) + (5. / 2.) * kb  #<==unit in eV/K
        
        #Rotation contributes to entropy: 
        #S(T) =( U(T) - U(0) ) / T + klnQ        
        qrot = self._rotational_partition_function(temperature)
        if self.geom == 'linear' : 
            S_r = kb + kb * np.log(qrot)
        elif  self.geom == 'nonlinear':
            S_r = (3./2.) * kb + kb * np.log(qrot)  
        else:
            S_r = 0.0

        #Vibration contributes to entropy:
        S_v = self._vibrational_entropy_contribution(temperature)
        
        #Electronic entropy (Boltzmann formula):
        #S = kBlnW
        S_e = kb * np.log(2 * self.spin + 1)
        
        S = (S_t + S_r + S_v + S_e) 
        return S

    def get_gibbs_energy(self, temperature, pressure):
        """Returns G(T), in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""
        H = self.get_enthalpy(temperature)
        S = self.get_entropy(temperature, pressure)
        G = H - temperature * S
        return G

    def get_helmholtz_energy(self, temperature, pressure):
        """Returns A(T), in eV, in the ideal gas
        approximation at a specified temperature (K)."""
        U = self.get_internal_energy(temperature)
        S = self.get_entropy(temperature, pressure)
        A = U - temperature * S
        return A
    
    
    def get_heat_capacity(self, temperature):
      
        Cv_v = self._vibrational_heat_capacity_contribution(temperature)
        
        if self.geom == 'nonlinear':
            Cv_t = (3. / 2.) * kb
            Cv_r = (3. / 2.) * kb
        elif self.geom == 'linear':
            Cv_t = (3. / 2.) * kb
            Cv_r =   kb
        else:# geom == 'monatomic':
            Cv_t = (3. / 2.) * kb
            Cv_r =   0
            
        Cv = Cv_t + Cv_r + Cv_v   
        return Cv

class HarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Often used for
    adsorbates.
    """
    #Codes from ASE
    
    def __init__(self, outcar='OUTCAR'):
        self.atoms  = vio.get_structures(outcar, mode='all')[0] #initial structure
        self.etotal = vio.get_energy(outcar, mode='all')[0]
        self.vibinfo = [line for line in vio.extra_vibinfo(outcar)if 'f' in line]
        self.vib_energies = [0.001 * vib for vib in vio.get_freqs(self.vibinfo, 'meV') if vib > 0]
        self.PV = 0 
 
    def get_internal_energy(self, temperature):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""
        U_0 = self.etotal
        zpe = self.get_ZPE_correction()
        U_v = self._vibrational_energy_contribution(temperature)
        U = U_0 + zpe + U_v
        return U

    def get_entropy(self, temperature):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""
        S = 0.
        S_v = self._vibrational_entropy_contribution(temperature)
        S += S_v
        return S

    def get_helmholtz_energy(self, temperature):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""
        U = self.get_internal_energy(temperature)
        S = self.get_entropy(temperature)
        A = U - temperature * S
        return A            

    def get_enthalpy(self, temperature):
        """Returns H(T), in eV, in the ideal gas approximation
        at a specified temperature (K)."""    
        U = self.get_internal_energy(temperature)
        H = U + self.PV
        return H
    
    def get_gibbs_energy(self, temperature):
        """Returns G(T), in eV, in the ideal gas approximation
        at a specified temperature (K)."""  
        #G = H - TS = U + PV -TS
        U = self.get_internal_energy(temperature)
        S = self.get_entropy(temperature)
        G = U - temperature * S + self.PV 
        return G    
    
    def get_heat_capacity(self, temperature):
        Cv = self._vibrational_heat_capacity_contribution(temperature)
        return Cv    
    
    