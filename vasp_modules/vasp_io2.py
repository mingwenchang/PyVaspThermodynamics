#!/usr/bin/env python3
# coding=UTF-8
"""
NAME
        vasp_io2.py -  This is the central module for 
                       post-vasp-calculation analysis
                       and gagcmc scripts.  
                           
 
DESCRIPTION
        Extract information from POSCAR/CONTCAR and OUTCAR, including 
        ionic positions, energies, forces, vibration frequencies etc.
        
        The read ionic positions from OUTCAR or POSCAR will be stored
        in an Atoms object. Geometric information or operations can 
        easily obtained or proceeded by the object methods.
        
            
DEVELOPER: 
    
    Dr. Ming-Wen Chang
    E-mail: ming.wen.c@gmail.com

"""

import os, sys
sys.path.append('%s/%s'% (os.getcwd(), 'modules'))


import numpy as np
import vasp_modules.assister as ast
import vasp_modules.data as data
from collections import OrderedDict
    
class Atoms:
    def __init__(self, atomtypes=None, natoms=None, positions=None,
                 cell=None, constraints=None):
        """
        Parameters:
            
        atomtypes: chemical symbols of atoms. Can be a string, a list of 
        chemical symbols, or a list of Atom objects.
        
        natoms: number of atoms per atomic species (one number for each atomic
        speices). A list of int. the length of natoms should be
        equal to the length of atomtypes
        
        positions: list of xyz-positions or anything that can be converted to 
        an ndarray of shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2),...].
        
        constraints: Anything that can be converted to an ndarray of shape 
        (n, 3). For vasp, it will do: [('T','T','T'), ('T','T','T'),...].
        
        cell: a 3x3 matrix  
 
        """
        
        self._names = ['atomtypes', 'natoms', 'positions', 'cell', 'constraints']
        
        if isinstance (atomtypes, (list, tuple)) and len(atomtypes) > 0 \
           and isinstance(atomtypes[0], Atoms) :
            """[Atoms1, Atoms2, ...]"""
            atomsobjs = list(atomtypes)
        elif isinstance (atomtypes, Atoms):
             """An Atoms obj"""
             atomsobjs = [atomtypes]
        else:
            """str1 or [str1, str2,..] """
            atomsobjs = None

        if atomsobjs is not None:
            param = atomsobjs[0].get_attribute(self._names)
            atoms = self.__class__(*param)
            others = atomsobjs[1:]
            for other in others:
                atoms.append(other)
            values = atoms.get_attribute(self._names) 
        else:
            if atomtypes is None:
                atomtypes = 'X'
                natoms = 1
                positions = np.array([[0.00, 0.00, 0.00]])
            values = [atomtypes, natoms, positions, cell, constraints]
        
        for i, name in enumerate(self._names):
            self.set_attribute(name, values[i])
           
    def get_attribute(self, name=None):
        """Get an attribute according to the name """   
        if name is None:
            names = self._names
            return [self.get_attribute(name) for name in names]
        
        if isinstance(name, (list, tuple)):
            names = name
            return [self.get_attribute(name) for name in names]
        
        if name == 'atomtypes':
            return self.atomtypes
        
        if name == 'natoms':
            return self.natoms
        
        if name == 'positions':
            return self.positions
        
        if name == 'constraints':
            if self.get_sdyn():
                return self.constraints
            else:
                return None
        if name == 'cell':
            if self.get_pbc():
                return self.cell
            else:
                return None
    
    def set_attribute(self, name, value):
        """Get an attribute according to the name"""
        if name == 'atomtypes':
            self.atomtypes = value    
        elif name == 'natoms':
            self.natoms = value
        elif name == 'positions':
            self.positions = value
        elif name == 'constraints':
            self.constraints = value  
        elif name == 'cell':
            self.cell = value
     
    """set atomic types"""   
    def set_atomic_types(self, atomtypes):
        if isinstance (atomtypes, str):
            self._atomtypes = [atomtypes]
        elif isinstance (atomtypes, (tuple, list)):
            self._atomtypes = list(atomtypes)
        else:
            raise ValueError
   
    def get_atomic_types(self):
        return self._atomtypes
    
    atomtypes = property(get_atomic_types, set_atomic_types)
 
    """set number of atoms for each chemcial species"""   
    def set_number_of_atoms(self, natoms):
        if isinstance (natoms, int):
            self._natoms = [natoms]
        elif isinstance (natoms, (tuple, list)):
            self._natoms = list(natoms)
        else:
            raise ValueError
            
    def get_number_of_atoms(self):
        return self._natoms 
    
    natoms = property(get_number_of_atoms, set_number_of_atoms)
    
    """set atomic positions"""   
    def set_atomic_positions(self, positions):
        ntotal =  self.get_total_atoms()
        positions = np.array(positions)
        self._positions = positions.reshape(ntotal,3)
            
    def get_atomic_positions(self):
        return self._positions 
    
    positions = property(get_atomic_positions, set_atomic_positions)
    
    """set atomic constraints"""  
    def set_atomic_constraints(self, constraints=None):
        if constraints is not None:
            self._sdyn = True
            if isinstance(constraints, (list, tuple, np.ndarray)):
                constraints = np.array(constraints)
            else:
                ntotal =  self.get_total_atoms()
                constraints = np.full((ntotal, 3), constraints)
        else:
            self._sdyn = False
            ntotal =  self.get_total_atoms()
            constraints = np.full((ntotal, 3), None)
        self._constraints = constraints
                
    def get_atomic_constraints(self):
        return self._constraints
    
    def del_atomic_constraints(self):
        self.set_atomic_constraints(constraints=None)
        
    constraints = property(get_atomic_constraints,
                           set_atomic_constraints,
                           del_atomic_constraints)
    
    """set cell"""  
    def set_cell(self, cell):
        if cell is not None and isinstance(cell, (list, tuple, np.ndarray)) :
            self._pbc = True
            self._cell = np.array(cell)
        else:
            self._pbc = False
            self._cell = np.full((3, 3), None) 
        
    def get_cell(self):
        return self._cell
    
    def del_cell(self):
        self.set_cell(self, None)
               
    cell = property(get_cell, set_cell, del_cell)
    
    #Get cell properties
    def get_cell_volume(self):
        return ast.tripleproduct(self._cell[0], self._cell[1], self._cell[2])
    
    def get_cell_lengths(self):
        a_norm  = ast.vectornorm(self._cell[0])
        b_norm  = ast.vectornorm(self._cell[1])
        c_norm  = ast.vectornorm(self._cell[2])
        return a_norm, b_norm, c_norm 
        
    def get_cell_angles(self):
        alpha = ast.angle(self._cell[1], self._cell[2])
        beta  = ast.angle(self._cell[0], self._cell[2])
        gamma = ast.angle(self._cell[0], self._cell[1])
        return alpha, beta, gamma
        
    #dictionalize   
    def get_dict_atomtypes(self):
        keys = self._atomtypes
        values = self._natoms
        return OrderedDict(zip(keys, values))
    
    def get_dict_positions(self):
        refitems = self.get_dict_atomtypes()
        return ast.match(self._positions, refitems)
    
    def get_dict_constraints(self):
        refitems = self.get_dict_atomtypes()
        return ast.match(self._constraints, refitems)
    
    #Get properties
    def get_pbc(self):
        return self._pbc
    
    def get_sdyn(self):
        return self._sdyn
    
    def get_stru_center(self):
        return ast.get_center_point(self._positions)
    
    def get_cell_center(self):
        if self.get_pbc():        
            cc = frac_to_cart(self.cell[0], 
                              self.cell[1], 
                              self.cell[2], 
                              np.array([[0.50, 0.50, 0.50]])) 
        else:
            cc = np.array([[0.00, 0.00, 0.00]])
        return cc 
    
    def get_total_atoms(self):
        return sum(self._natoms)
    
    def get_chemical_formula(self):
        #chemical formula
        cf = '' 
        for atom, number in zip(self.atomtypes, self.natoms):
            cf +=atom
            if number > 1:
                cf +=str(number)
        return cf
    
    def get_atomic_masses(self):
        masses = []
        for atom, number in zip(self.atomtypes, self.natoms):
            masses += number * [data.atomic_masses[atom]]
        return masses 
    
    def get_molecular_mass(self):
        m = sum(self.get_atomic_masses())
        return m #unit in amu
    
    def get_center_of_mass(self):
        masses = np.array(self.get_atomic_masses())
        positions = self.get_atomic_positions()
        com = np.dot(masses, positions) / sum(masses)
        return com #unit in amu
            
    def get_moments_of_inertia(self):
        """Get the moments of inertia along the principal axes.

        The three principal moments of inertia are computed from the
        eigenvalues of the symmetric inertial tensor. Periodic boundary
        conditions are ignored. Units of the moments of inertia are
        amu*angstrom**2.
        
        Following codes are from ASE module:
            
        """
        
        com = self.get_center_of_mass()
        positions = self.get_atomic_positions()
        positions -= com  # translate center of mass to the center of mass
        masses = np.array(self.get_atomic_masses())

        # Initialize elements of the inertial tensor
        I11 = I22 = I33 = I12 = I13 = I23 = 0.0
        for i in range(len(self)):
            x, y, z = positions[i]
            m = masses[i]

            I11 += m * (y ** 2 + z ** 2)
            I22 += m * (x ** 2 + z ** 2)
            I33 += m * (x ** 2 + y ** 2)
            I12 += -m * x * y
            I13 += -m * x * z
            I23 += -m * y * z

        I = np.array([[I11, I12, I13],
                      [I12, I22, I23],
                      [I13, I23, I33]])

        evals, evecs = np.linalg.eigh(I)
        return evals    
    
    def get_distance_matrix(self):
        return ast.get_distance_matrix(self._positions)    
                
    def get_fractional(self, pos=None):
        if pos is None:
            pos = self.positions
        frac = cart_to_frac(self.cell[0], self.cell[1], self.cell[2], pos)
        return frac
    
    def truncate(self, atomtypes, natoms=None, mode='tail'):
        #dictionalizeation
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
        
        if isinstance(atomtypes, str):
            atomtypes = [atomtypes]
            
        if natoms is None:
            natoms = [len(dpos[atom]) for atom in  atomtypes]
        elif isinstance(natoms, int):
            natoms = [natoms]
       
        positions = np.empty(shape=(0,3))
        constraints = np.empty(shape=(0,3))
        for atom, num in zip(atomtypes, natoms):
            if mode[0].lower() == 't':
                pos= dpos[atom][-num:]
                con = dcon[atom][-num:]
            elif mode[0].lower() == 'h':
                pos = dpos[atom][:num]
                con = dcon[atom][:num]
                
            positions = np.append(positions, pos, axis=0 )
            constraints = np.append(constraints, con, axis=0 )
            
        if not self.get_sdyn():
            constraints = None
            
        if  self.get_pbc():
            cell = self.cell
        else:
            cell = None
        
        atomsobj = self.__class__(atomtypes, natoms, positions, cell, constraints)
        return atomsobj
        
    def append(self, other):
        """Extend an atoms object by appending other atoms object"""
        
        #dictionalization for self
        dpos1 = self.get_dict_positions() 
        dcon1 = self.get_dict_constraints()
        
        #dictionalization for other
        dpos2 = other.get_dict_positions() 
        dcon2 = other.get_dict_constraints()
        
        #Combination two atoms objects 
        dpos = ast.combine(dpos1, dpos2) 
        dcon = ast.combine(dcon1, dcon2)
        
        #
        datomtypes = ast.pair_key_and_amount(dpos)
        atomtypes = list(datomtypes.keys())
        natoms = list(datomtypes.values())
        positions = ast.merge(dpos)
        
        #Get sdyn 
        sdyn1 = self.get_sdyn()
        sdyn2 = other.get_sdyn()
        if sdyn1 and sdyn2:
            constraints = ast.merge(dcon)
        else:
            constraints = None
        
        #Get PBC
        pbc1 = self.get_pbc()
        pbc2 = other.get_pbc()
        if pbc1 or pbc2:
            cell = self.cell
        else:
            cell = None
        
        #Update information
        self.atomtypes = atomtypes
        self.natoms = natoms
        self.positions = positions 
        self.constraints = constraints
        self.cell = cell
        
    def pop(self, atom, i=-1):
        """ Remove a set of 'X' atoms according to the indices""" 
        
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
        
        if i is None:
            dpos[atom] = []
            dcon[atom] = []
        else:
            dpos[atom] = np.delete(dpos[atom], i, axis=0)
            dcon[atom] = np.delete(dcon[atom], i, axis=0)
        
        if len(dpos[atom]) == 0:
            del dpos[atom]
            del dcon[atom]
        
        #Update information
        datomtypes = ast.pair_key_and_amount(dpos)
        self.atomtypes = list(datomtypes.keys())
        self.natoms = list(datomtypes.values())
        self.positions = ast.merge(dpos)   
        self.constraints = ast.merge(dcon)   
    
    def grab(self, atom, i=None):
        """ grab a set of 'X' atoms according to the indices""" 
        
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
        
        if i is None:
            number = len(dpos[atom])
            i = list(range(number))
        elif isinstance(i, int):
            i = [i]
        
        positions = np.take(dpos[atom], i, axis=0)
        constraints = np.take(dcon[atom], i, axis=0)

        atomtypes = atom
        natoms = len(positions)
        
        if not self.get_sdyn():
            constraints = None
            
        if  self.get_pbc():
            cell = self.cell
        else:
            cell = None
        
        atomsobj = self.__class__(atomtypes, natoms, positions, cell, constraints)
        return atomsobj
        
    def sort (self, point=None):
        """sort atoms using the relative distances between atoms 
           and a specific point.
           
           The defalut point is the center of the current structure 
        """
        
        if point is None:
            point = self.get_stru_center()
        
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
      
        for atom in self.atomtypes:
            refdists = []
            for pos in dpos[atom]:
                dist = ast.distance(pos, point)
                refdists.append(dist)    
            dpos[atom] = np.array(ast.sort_two_lists(refdists, dpos[atom])[1])
            dcon[atom] = np.array(ast.sort_two_lists(refdists, dcon[atom])[1])
        
        self.positions = ast.merge(dpos)   
        self.constraints = ast.merge(dcon)  
        
    #Manipulations
    def move_to_origin(self):
        """Set the center of a structure to (0.00, 0.00, 0.00)."""   
        self.positions = ast.move_to_origin(self.positions)
        
    def move_to_cell_center(self):
        if self._pbc:
            cc = self.get_cell_center()
            self.positions = ast.move_to_the_point(self.positions, cc)
            pass
        else:
            self.move_to_origin()
            
    def move_to_the_point(self, point):
        self.positions = ast.move_to_the_point(self.positions, point)
        
    def rotate(self, angle=None, axis=None):
        if angle is None:
            angle   = np.random.uniform(0,360)
        if axis is None:
            axis = np.random.choice(['x', 'y', 'z'])
            
        center = self.get_stru_center() 
        self.move_to_origin()
        self.positions = ast.rotate_structure(self.positions, angle, axis)     
        self.move_to_the_point(center)
        
    def euler_rotate(self, phi=None, theta=None, psi=None):
        if phi is None:
            phi   = np.random.uniform(0,360)
        if theta is None:
            theta   = np.random.uniform(0,180)
        if psi is None:
            psi   = np.random.uniform(0,360)
    
        center = self.get_stru_center() 
        self.move_to_origin()
        self.positions = ast. euler_rotate(self.positions, phi, theta, psi)     
        self.move_to_the_point(center)
                
    def rattle(self, ratio=1.00, delta=0.50, seed=None):
        """Randomly displace atoms.
        
        The displacement matrix is generated from a Gaussian distribution.
        
        delta: Standard deviation (spread or “width”) of the distribution.

        """
        
        ntotal = self.get_total_atoms()
        rs = np.random.RandomState(seed)
        rdm = rs.normal(scale=delta, size=(ntotal,3))
    
        #Selected atoms randomly 
        nmoves = int(ntotal * ratio)
        if nmoves < 1:
            nmoves = 1
        select = ast.selector(nmoves, ntotal)  

        for i in select:
            self.positions[i] += rdm[i]
        
    def sprain(self, ratio=0.50):
        """Randomly selected atoms and rotate them."""
        center = self.get_stru_center() 
        ntotal = self.get_total_atoms()
        nmoves = int(ntotal * ratio)
        
        if nmoves < 1:
            nmoves = 1
        
        self.move_to_origin()
        indices = ast.selector(nmoves, ntotal)  
        selected = np.take(self.positions, indices, axis=0)
        
        phi   = np.random.uniform(0,360)
        theta   = np.random.uniform(0,180)
        psi   = np.random.uniform(0,360)
        positions = ast. euler_rotate(selected, phi, theta, psi)  
        
        for i, j in enumerate(indices):
            self.positions[j] = positions[i]
            
        self.move_to_the_point(center)
            
    def twist(self):
        """split structure into two groups then rotate them randomly"""
        h1, h2 = self.split()
        h1.euler_rotate(); h2.euler_rotate()
        atoms = self.__class__([h1, h2])
        values = atoms.get_attribute()

        for i, name in enumerate(self._names):
            self.set_attribute(name, values[i])
        
    def permutate(self, ratio=0.50):
        """Randomly exchange atoms."""
        dpos = self.get_dict_positions() 
        dcon = self.get_dict_constraints()
        
        if len(self.atomtypes) > 1 :
            i, j = ast.selector(2, len(self.atomtypes))
        else:
            i , j = 0, 0
        
        atom1 = self.atomtypes[i]; atom2 = self.atomtypes[j]
        natoms1 = self.natoms[i]; natoms2 = self.natoms[j]
    
        if natoms1 < natoms2:
            nexchanges = int(natoms1 * ratio)   
            if nexchanges < 1:
                nexchanges = 1 
        else:
            nexchanges = int(natoms2 * ratio) 
            if nexchanges < 1:
                nexchanges = 1
        sel1 = ast.selector(nexchanges, natoms1)
        sel2 = ast.selector(nexchanges, natoms2)
        
        for k, l  in zip(sel1, sel2):
            dpos[atom1][k], dpos[atom2][l] = np.copy(dpos[atom2][l]), np.copy(dpos[atom1][k])
            dcon[atom1][k], dcon[atom2][l] = np.copy(dcon[atom2][l]), np.copy(dcon[atom1][k])

        self.positions = ast.merge(dpos)   
        self.constraints = ast.merge(dcon)  
        
    def split(self, normvect=None):
        """Split an Atoms obj into two Atoms objs."""
        if normvect is None:
            normvect = ast.generate_a_normvect()
            
        other = self.copy()
        center = other.get_stru_center()
        
        other.move_to_origin()
        dpos = other.get_dict_positions()
        dcon = other.get_dict_constraints()
        datomtyps = other.get_dict_atomtypes()
        
        rdpos = OrderedDict(); ldpos = OrderedDict()
        rdcon = OrderedDict(); ldcon = OrderedDict()
        
        for atom in datomtyps.keys():
            rdpos[atom] = []; ldpos[atom] = []
            rdcon[atom] = []; ldcon[atom] = []
            natom = datomtyps[atom]
            for i in range(natom):
                pos = dpos[atom][i]
                con = dcon[atom][i]
                if ast.angle(pos, normvect) < 90:
                    rdpos[atom].append(pos)
                    rdcon[atom].append(con)
                else:
                    ldpos[atom].append(pos)
                    ldcon[atom].append(con)
                    
            if len(rdpos[atom]) > 0:
                rdpos[atom] = ast.move_structure(rdpos[atom], center)
            else:
                del rdpos[atom]
                del rdcon[atom]
        
            if len(ldpos[atom]) > 0:
                ldpos[atom] = ast.move_structure(ldpos[atom], center)
            else:
                del ldpos[atom] 
                del ldcon[atom]
                
        rdictatoms = ast.pair_key_and_amount(rdpos)
        ratomtypes = list(rdictatoms.keys())
        rnatoms = list(rdictatoms.values())       
        rpos = ast.merge(rdpos)
        
        ldictatoms = ast.pair_key_and_amount(ldpos)
        latomtypes = list(ldictatoms.keys())
        lnatoms = list(ldictatoms.values())       
        lpos = ast.merge(ldpos) 
        
        if self.get_pbc():
            cell= self.cell
        else:
            cell = None
            
        if self.get_sdyn():
            rcon = ast.merge(rdcon)
            lcon = ast.merge(ldcon)
        else:
            rcon = None
            lcon = None
            
        ratoms = self.__class__(ratomtypes, rnatoms, rpos, cell, rcon)
        ratoms.sort(center)
        latoms = self.__class__(latomtypes, lnatoms, lpos, cell, lcon)
        latoms.sort(center)
            
        return ratoms, latoms
    
    
    def copy(self):
        """Return a copy"""
        atomsobj = self.__class__(self.atomtypes, self.natoms, self.positions, self.cell, self.constraints)
        return atomsobj
    
    def write(self, filename=None, format='xyz'):
        """Write """
        if filename is None:
            tag = self.get_chemical_formula()
        else:
            tag = filename
            
        if format == 'xyz':
            if '.xyz' not in tag:
                tag += '.xyz'
            write_xyz(self, filename=tag)
        elif format == 'vasp':
            if  'POSCAR' not in tag and 'CONTCAR' not in tag:
                tag += '.vasp'
            write_poscar(self, filename=tag)

    def __repr__(self):
        #chemical formula
        cf = self.get_chemical_formula()
        s = "Atoms('%s')" % (cf) 
        return s
    
    def __add__(self, other):        
        objs = [self.copy(), other]
        atomsobj = self.__class__(objs)
        return atomsobj  
    
    def __len__(self):
        return len(self.positions)
        
        
def read_poscar(filename='POSCAR'):
    
    if os.path.exists(filename):
        f = open(filename, 'r')
    else:
        print (filename, "doesn't exit")
    
    # A comment line    
    comment = f.readline() 
    
    #A scaling factor/the lattice constant
    lc = float(f.readline().split()[0])
    
    #Lattice vectors     
    cell = [ ]
    for i in range(3):
        l = f.readline().split()
        vect = float(l[0]), float(l[1]), float(l[2]) 
        cell.append(vect)
    cell = np.array(cell) * lc
    
    #Get atomic types
    atomtypes = [i for i in f.readline().split()] 
    try:
        int(atomtypes[0])
        atomamounts = [int(i) for i in atomtypes]
        atomtypes = comment.split()
    except ValueError:
        atomamounts = [int(i) for i in f.readline().split()] 
    
    #Check selective dynamics tag 
    sdyn_line = f.readline().strip()
    if sdyn_line[0].upper() == 'S':
        sdyn = True
        format_line = f.readline().strip()
    else:
        sdyn = False
        format_line = sdyn_line
    
    #Check formate
    if format_line[0].upper() == 'C' or format_line[0].upper() == 'K':
        cartesian = True
    else:
        cartesian = False 
        
    #Read coordinates, constraints
    total_atoms = sum(atomamounts)
    if sdyn:
        positions = []
        constraints = []  
        for i in range(total_atoms):
            l = f.readline().split()
            vect = float(l[0]), float(l[1]), float(l[2])
            positions.append(vect)
            const = l[3], l[4], l[5]
            constraints.append(const)
        positions = np.array(positions)
        constraints = np.array(constraints)
    else:
        positions = []
        constraints = None
        for i in range(total_atoms):
            l = f.readline().split()
            vect = float(l[0]), float(l[1]), float(l[2])
            positions.append(vect)
        positions = np.array(positions)
            
    #Convert Fractional to Cartesian
    if not cartesian:
        positions = frac_to_cart(cell[0], cell[1], cell[2], positions)
        
    f.close()
    
    atoms = Atoms(atomtypes, atomamounts, positions, cell, constraints)

    return atoms
            
#Convert a set of position vectors to Cartesian from Fractional 
def frac_to_cart(v1, v2, v3, posvects):
    tmatrix = ast.build_tmatrix(v1,v2,v3,'f2c')
    cartesian = np.dot(posvects, tmatrix.T)
    return  cartesian

#Convert a set of position vectors to Fractional from Cartesian
def cart_to_frac(v1, v2, v3, posvects):
    tmatrix = ast.build_tmatrix(v1,v2,v3,'c2f')
    fractional = np.dot(posvects, tmatrix.T)
    return fractional

#Write xyz file
def write_xyz(obj, filename='POSCAR.xyz'):
    if isinstance(obj, Atoms):
        dictpos = obj.get_dict_positions()
    else:
        dictpos = obj
        
    stdout = sys.stdout
    xyz = open(filename, 'w')
    sys.stdout = xyz
    
    dictatoms = ast.pair_key_and_amount(dictpos)
    atomtypes = list(dictatoms.keys())
    natoms = list(dictatoms.values())
    ntotal = sum(natoms)
    
    print(ntotal, end='\n')
    print(atomtypes, end='\n')
       
    for atom in atomtypes:
        for pos in dictpos[atom]: 
            print (" %s %12.6f %12.6f %12.6f"%(atom, pos[0], pos[1], pos[2]), end='\n')
    sys.stdout = stdout    
    xyz.close()

#Write POSCAR file
def write_poscar(obj, cell=None, dictcon=None, filename='POSCAR.vasp', tag='Cartesian', ver='vasp5'):
    if isinstance(obj, Atoms):
        sdyn = obj.get_sdyn()
        cell = obj.get_cell()
        dictpos = obj.get_dict_positions()
        dictcon = obj.get_dict_constraints()
    else:
        dictpos = obj
        if dictcon is None:
            sdyn= False
        else:
            sdyn= True
        
    stdout = sys.stdout
    poscar = open(filename, 'w')
    sys.stdout = poscar
    
    dictatoms = ast.pair_key_and_amount(dictpos)
    atomtypes = list(dictatoms.keys())
    natoms = list(dictatoms.values())
    
    #write title section
    print (' '.join(atomtypes), end='\n')
    
    #write 1.00 as the scaling factor 
    print ('1.00', end='\n')

    #write unscaling lattice vectors
    for i in range(3):
        print (" %18.15f   %18.15f   %18.15f" %(cell[i][0], cell[i][1], cell[i][2]), end='\n')
        
    #write atom type 
    if ver == 'vasp5':
        print ('   '.join(atomtypes), end='\n')
        
    #Write the number of atoms        
    print ('   '.join(map(str, natoms)), end ='\n')

    if sdyn:
        print ('Selective dynamics', end='\n')
        
    if  tag.upper()[0] == 'C' or tag.upper()[0] == 'K':
        print ('Cartesian', end='\n')
    elif tag.upper()[0] == 'D':
        print ('Direct', end='\n')

    #write coordinates and constrains of atoms
    for atom in atomtypes:
        if sdyn:
            for pos, cons in zip(dictpos[atom], dictcon[atom]):
                print (" %18.15f %18.15f %18.15f  %s"\
                    %(pos[0], pos[1], pos[2], '    '.join(cons)), end='\n')
        else:
            for pos in dictpos[atom]:
                print (" %18.15f %18.15f %18.15f" %(pos[0], pos[1], pos[2]), end='\n')
            
    sys.stdout = stdout
    poscar.close()
    
    
#Functions for reading OUTCAR   
def get_ionic_types(file='OUTCAR'):
    with open(file) as outcar:
        for line in outcar:
            if line.find(' POSCAR =') > -1:
                atomtypes = line.split()[2:]
    return atomtypes       
        
def get_number_of_ions_per_type(file='OUTCAR'):
    with open(file) as outcar:
        for line in outcar:
            if line.find('ions per type' ) > -1:
                nions = [int(i) for i in line.split()[4:]]
    return nions   
    
def get_lattice_vectors(file='OUTCAR'):
    with open(file) as outcar:    
        start = False
        n = 0 
        vectors = [ ]
        for line in outcar:
            if line.find('direct lattice vectors') > -1:
                start = True
                continue
            if start:
                vectors.append([float(i) for i in line.split()[0:3]])
                n += 1 
            if n >=3:
                break
    return np.array(vectors)
 
def get_structures(file='OUTCAR', mode=None):
    iontypes = get_ionic_types(file)
    nions = get_number_of_ions_per_type(file)
    cell = get_lattice_vectors(file)
    with open(file) as outcar:    
        start = False
        n = 0 
        strus = []
        positions = []
        for line in outcar:
            if line.find('position of ions in cartesian coordinates') > -1 or \
               line.find('TOTAL-FORCE (eV/Angst)') > -1:
                start = True
                continue
            if start and line.find('--------------') == -1:
                positions.append([float(i) for i in line.split()[0:3]])
                n += 1
            if n >= sum(nions):
                atomsobj = Atoms(iontypes, nions, positions, cell)
                strus.append(atomsobj)   
                start = False
                n = 0
                positions = []
                
    if mode is None:
        strus = strus[-1]
                     
    return strus


#Get DFT energy from OUTCAR
def get_energy(filename = 'OUTCAR', mode=None):
    if mode is None:
        ezero = 999999
        #efree = 999999
    else:
        ezero =  [ ]
        #efree =  [ ]
        
    if  os.path.exists(filename):
        for line in open(filename, 'r'):
            #energy(sigma->0)
            if line.startswith('  energy  without entropy'):
                if mode is None:
                    ezero = float(line.split()[-1])
                else:
                    ezero.append(float(line.split()[-1]))
            """       
            #free energy
            if line.lower().startswith('  free  energy   toten'):
                if mode is None:
                    efree = float(line.split()[-2])
                else:
                    efree.append(float(line.split()[-2]))
            """ 
    else:
        print (filename, ' was not found')
        ezero = 999999
    return ezero
        
#Get Maximum Force from OUTCAR
def get_force(filename='OUTCAR', mode=None):
    if mode is None:
        force = 999999
    else:
        force =  [ ]

    if os.path.exists(filename):
        for line in open(filename, 'r'):
            #FORCES: max atom
            if line.startswith('  FORCES: max atom, RMS'): 
                if mode is None:
                    force = float(line.split()[-2])
                else:
                    force.append(float(line.split()[-2]))
    else:
        print (filename, ' was not found')
        force = 99999
    return force


#Get Eigenvectors and eigenvalues of the dynamical matrix from OUTCAR    
def extra_vibinfo(filename='OUTCAR'):
    if os.path.exists(filename):
        outcar = open(filename,'r')
        start = 'Eigenvectors and eigenvalues of the dynamical matrix'
        end   = 'Finite differences POTIM' 
        sqrt  = 'Eigenvectors after division by SQRT(mass)'
    else:
         raise IOError('%s does not exist!!' %(filename))
        
    infomatrix = []
    switch = False
    for line in outcar:
        line = line.strip()
        if start in line:
             switch = True
        elif sqrt in line:
            switch = None
        elif end in line:
             switch = False
        
        if switch and line !=  '':
               infomatrix.append(line)
        elif switch == None:
            infomatrix = []
        else:
            continue
    infomatrix = infomatrix[2:]
    return infomatrix

#Get frequencies
def get_freqs(infomatrix, unit='cm-1'):
    freqs = []
    freqinfo = [line for line in infomatrix if 'cm-1' in line]
    for line in freqinfo:
        mode = line.split('=')[0] #i.e. 1 f or 1 f/i
        values = line.split('=')[1]  
        
        if unit.lower()=='mev':
            freq = float(values.split()[6])   
        else:
            freq = float(values.split()[4])  #unit in cm-1
    
        if 'f/i' in mode:
            freq = -1 * freq
            
        freqs.append(freq)
 
    return freqs

#Get Eigenvectors of modes 
def get_dymatrix(infomatrix):
    dymatrix = OrderedDict()
    for line in infomatrix:
        if 'X' in line:
            continue
        elif 'f' in line:
            mode = line.split('=')[0].strip().replace(' ','')
            mode = mode.replace('f/i','i')
            dymatrix[mode] = []
        else:
            vector = [float(pos) for pos in line.split()[3:]]
            dymatrix[mode].append(vector)
    return dymatrix    

#whether is a successful calculation
def is_a_successful_vasp_job():
    string = 'General timing and accounting informations for this job:'
    info = ast.grep_a_string(string, 'OUTCAR')
    
    if len(info) > 0:
        return True
    else:
        return False
