# PyVaspThermodynamics
Calculate thermodynamic properties from VASP outputs 

# Run
- for a gas molecule: ./idealthermo.py OUTCAR 
- for an adsorbed molecule: ./surfthermo.py OUTCAR
- for a surface reaction: ./surfrxn.py OUTCAR.IS OUTCAR.TS OUTCAR.FS

# Outputs: 
- thermo.dat: thermodynamic data, total energy, enthalpy, entropy, free energy, etc. 
- zpe.dat: zero-potential energy
- rxn.dat: Gibbs free energy of the reaction 

