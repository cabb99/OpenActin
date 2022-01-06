#!/home/cab22/miniconda3/bin/python

#SBATCH --account=commons
#SBATCH --export=All
#SBATCH --partition=commons
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --array=0-15
#SBATCH --mem=16G

import os
import subprocess
import itertools
import numpy as np
import warnings
import pandas
import time
import argparse

class SlurmJobArray():
    """ Selects a single condition from an array of parameters using the SLURM_ARRAY_TASK_ID environment variable. The parameters need to be supplied as a dictionary. if the task is not in a slurm environment, the test parameters will supersede the parameters, and the job_id would be taken as 0.  Example:
        parameters={"epsilon":[100],
                    "aligned":[True,False],
                    "actinLen":[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
                    "repetition":range(5),
                    "temperature":[300],
                    "system2D":[False],
                    "simulation_platform":["OpenCL"]}
        test_parameters={"simulation_platform":"CPU"}
        sjob=SlurmJobArray("ActinSimv6", parameters, test_parameters)
        :var test_run: Boolean: This simulation is a test
        :var job_id: SLURM_ARRAY_TASK_ID
        :var all_parameters: Parameters used to initialize the job
        :var parameters: Parameters for this particular job
        :var name: The name (and relative path) of the output
        
    """
    def __init__(self, name, parameters, test_parameters={},test_id=0):
        """
        Args:
            name:
            parameters:    
        Returns:
        
        name:
        parameters:
        """
        
        self.all_parameters=parameters
        self.test_parameters=test_parameters
        
        #Parse the slurm variables
        self.slurm_variables={}
        for key in os.environ:
            if len(key.split("_"))>1 and key.split("_")[0]=='SLURM':
                self.slurm_variables.update({key:os.environ[key]})
        
        #Check if there is a job id
        self.test_run=False
        try:
            self.job_id=int(self.slurm_variables["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            self.test_run=True
            warnings.warn("Test Run: SLURM_ARRAY_TASK_ID not in environment variables")
            self.job_id=test_id
            
        keys=parameters.keys()
        self.all_conditions=list(itertools.product(*[parameters[k] for k in keys]))
        self.parameter=dict(zip(keys,self.all_conditions[self.job_id]))
        
        #The name only includes enough information to differentiate the simulations.
        self.name=f"{name}_{self.job_id:03d}_" + '_'.join([f"{a[0]}_{self[a]}" for a in self.parameter if len(self.all_parameters[a])>1])
                
    def __getitem__(self, name):
        if self.test_run:
            try:
                return self.test_parameters[name]
            except KeyError:
                return self.parameter[name]
        else:
            return self.parameter[name]	
        
    def __getattr__(self, name: str):
        """ The keys of the parameters can be called as attributes
        """
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        elif name in self.parameter:
            return self[name]
        else:
            return object.__getattribute__(self, name)

    def __repr__(self):
        return str(self.parameter)
        
    def keys(self):
        return str(self.parameters.keys())

    def print_parameters(self):
        print(f"Number of conditions: {len(self.all_conditions)}")
        print("Running Conditions")
        for k in self.parameter.keys():
            print(f"{k} :", f"{self[k]}")
        print()
        
    def print_slurm_variables(self):
        print("Slurm Variables")
        for key in self.slurm_variables:
            print (key,":",self.slurm_variables[key])
        print()
            
    def write_csv(self, out=""):
        s=pandas.concat([pandas.Series(self.parameter), pandas.Series(self.slurm_variables)])
        s['test_run']=self.test_run
        s['date']=time.strftime("%Y_%m_%d")
        s['name']=self.name
        s['job_id']=self.job_id
        
        if out=='':
            s.to_csv(self.name+'.param')
        else:
            s.to_csv(out)

################
# Coarse Actin #
################

#!/usr/bin/python3
"""
Coarse Actin simulations using a custom
"""

import openmm
import openmm.app
from simtk import unit
import numpy as np
import pandas
import sklearn.decomposition
import configparser
import prody
import scipy.spatial.distance as sdist
import os
import sys

__author__ = 'Carlos Bueno'
__version__ = '0.2'
#__location__ = os.path.realpath(
#    os.path.join(os.getcwd(), os.path.dirname(__file__)))
#__location__="/scratch/cab22/Bundling/Persistence_length/Persistence_length"
__location__='.'
_ef = 1 * unit.kilocalorie / unit.kilojoule  # energy scaling factor
_df = 1 * unit.angstrom / unit.nanometer  # distance scaling factor
_af = 1 * unit.degree / unit.radian  # angle scaling factor


def parseConfigTable(config_section):
    """Parses a section of the configuration file as a table"""

    def readData(config_section, a):
        """Filters comments and returns values as a list"""
        temp = config_section.get(a).split('#')[0].split()
        l = []
        for val in temp:
            val = val.strip()
            try:
                x = int(val)
                l += [x]
            except ValueError:
                try:
                    y = float(val)
                    l += [y]
                except ValueError:
                    l += [val]
        return l

    data = []
    for a in config_section:
        if a == 'name':
            columns = readData(config_section, a)
        elif len(a) > 3 and a[:3] == 'row':
            data += [readData(config_section, a)]
        else:
            print(f'Unexpected row {readData(config_section, a)}')
    return pandas.DataFrame(data, columns=columns)


# Random rotation matrix


def random_rotation():
    """Generate a 3D random rotation matrix.
    Returns:
        np.matrix: A 3D rotation matrix.
    """
    x1, x2, x3 = np.random.rand(3)
    R = np.matrix([[np.cos(2 * np.pi * x1), np.sin(2 * np.pi * x1), 0],
                   [-np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                   [0, 0, 1]])
    v = np.matrix([[np.cos(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sin(2 * np.pi * x2) * np.sqrt(x3)],
                   [np.sqrt(1 - x3)]])
    H = np.eye(3) - 2 * v * v.T
    M = -H * R
    return M


# Optimal rotation matrix
# The longest coordinate is X, then Y, then Z.


def optimal_rotation(coords):
    c = coords.copy()
    c -= c.mean(axis=0)
    pca = sklearn.decomposition.PCA()
    pca.fit(c)
    # Change rotoinversion matrices to rotation matrices
    rot = pca.components_[[0, 1, 2]]
    if np.linalg.det(rot) < 0:
        rot = -rot
        #print(rot, np.linalg.det(rot))
    return rot


class SystemData:
    def __init__(self, atoms, bonds=None, angles=None, dihedrals=None, impropers=None):
        self.atoms = atoms
        self.atoms.index = np.arange(1, len(self.atoms) + 1)
        self.masses = atoms[['type', 'mass']].drop_duplicates()
        self.masses.index = np.arange(1, len(self.masses) + 1)
        self.n_atoms = len(self.atoms)
        self.n_atomtypes = len(self.masses)

        if bonds is not None:
            self.bonds = bonds
            self.bonds.index = np.arange(1, len(self.bonds) + 1)
            self.bondtypes = bonds[['type', 'x0', 'k']].drop_duplicates()
            self.bondtypes.index = np.arange(1, len(self.bondtypes) + 1)
            self.n_bonds = len(self.bonds)
            self.n_bondtypes = len(self.bondtypes)
        else:
            self.bonds = pandas.DataFrame()
            self.bondtypes = pandas.DataFrame()
            self.n_bonds = 0
            self.n_bondtypes = 0

        if angles is not None:
            self.angles = angles
            self.angles.index = np.arange(1, len(self.angles) + 1)
            self.angletypes = angles[['type', 'x0', 'k']].drop_duplicates()
            self.angletypes.index = np.arange(1, len(self.angletypes) + 1)
            self.n_angles = len(self.angles)
            self.n_angletypes = len(self.angletypes)
        else:
            self.angles = pandas.DataFrame()
            self.angletypes = pandas.DataFrame()
            self.n_angles = 0
            self.n_angletypes = 0

        if dihedrals is not None:
            self.dihedrals = dihedrals
            self.dihedrals.index = np.arange(1, len(self.dihedrals) + 1)
            self.dihedraltypes = dihedrals[['type', 'x0', 'k']].drop_duplicates()
            self.dihedraltypes.index = np.arange(1, len(self.dihedraltypes) + 1)
            self.n_dihedrals = len(self.dihedrals)
            self.n_dihedraltypes = len(self.dihedraltypes)
        else:
            self.dihedrals = pandas.DataFrame()
            self.dihedraltypes = pandas.DataFrame()
            self.n_dihedrals = 0
            self.n_dihedraltypes = 0

        if impropers is not None:
            self.impropers = impropers
            self.impropers.index = np.arange(1, len(self.impropers) + 1)
            self.impropertypes = impropers[['type', 'x0', 'k']].drop_duplicates()
            self.impropertypes.index = np.arange(1, len(self.impropertypes) + 1)
            self.n_impropers = len(self.impropers)
            self.n_impropertypes = len(self.impropertypes)
        else:
            self.impropers = pandas.DataFrame()
            self.impropertypes = pandas.DataFrame()
            self.n_impropers = 0
            self.n_impropertypes = 0

        # self.n_bonds=len(self.bonds)
        # self.n_bondtypes=len(self.bondtypes)
        # self.xmin,self.xmax=atoms['x'].min(),atoms['x'].max()
        # self.ymin,self.ymax=atoms['y'].min(),atoms['y'].max()
        # self.zmin,self.zmax=atoms['z'].min(),atoms['z'].max()

    def write_data(self, file_name='actin.data', box_size=1000):
        self.xmin, self.xmax = 0, box_size
        self.ymin, self.ymax = 0, box_size
        self.zmin, self.zmax = 0, box_size
        with open(file_name, 'w+') as f:
            f.write('LAMMPS data file generated with python\n\n')
            f.write('\t%i atoms\n' % self.n_atoms)
            f.write('\t%i  bonds\n' % self.n_bonds)
            f.write('\t%i  angles\n' % self.n_angles)
            f.write('\t%i  dihedrals\n' % self.n_dihedrals)
            f.write('\t%i  impropers\n' % self.n_impropers)
            f.write('\n')
            f.write('\t%i  atom types\n' % self.n_atomtypes)
            f.write('\t%i  bond types\n' % self.n_bondtypes)
            f.write('\t%i  angle types\n' % self.n_angletypes)
            f.write('\t%i  dihedral types\n' % self.n_dihedraltypes)
            f.write('\t%i  improper types\n' % self.n_impropertypes)
            f.write('\n')
            f.write('\t %f %f xlo xhi\n' % (self.xmin, self.xmax))
            f.write('\t %f %f ylo yhi\n' % (self.ymin, self.ymax))
            f.write('\t %f %f zlo zhi\n' % (self.zmin, self.zmax))
            f.write('\n')
            f.write('Masses\n\n')
            for i, m in self.masses.iterrows():
                f.write('\t%i\t%f\n' % (i, m.mass))
            f.write('\n')
            f.write('Atoms\n\n')
            for i, a in self.atoms.iterrows():
                f.write('\t%i\t%i\t%i\t%f\t%f\t%f\t%f\n' % (i, a.molecule, a.type, a.q, a.x, a.y, a.z))
            f.write('\n')
            if self.n_bonds > 0:
                f.write('Bonds\n\n')
                for i, b in self.bonds.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j))
                f.write('\n')
            if self.n_angles > 0:
                f.write('Angles\n\n')
                for i, b in self.angles.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l))
                f.write('\n')
            if self.n_dihedrals > 0:
                f.write('Dihedrals\n\n')
                for i, b in self.dihedrals.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l, b.m))
                f.write('\n')
            if self.n_impropers > 0:
                f.write('Impropers\n\n')
                for i, b in self.impropers.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l, b.m))
                f.write('\n')

    def write_pdb(self, file_name='actin.pdb'):
        import string
        cc = (string.ascii_uppercase.replace('X','') + string.ascii_lowercase + '1234567890'+'X')*1000
        cc_d = dict(zip(range(1, len(cc) + 1), cc))
        pdb_line = '%-6s%5i %-4s%1s%3s %1s%4i%1s   %8s%8s%8s%6.2f%6.2f          %2s%2s\n'
        pdb_atoms = self.atoms.copy()
        pdb_atoms['serial'] = np.arange(1, len(self.atoms) + 1)
        # pdb_atoms['name']       = self.atoms['type'].replace({1:'A1',2:'A2',3:'A3',4:'A4',5:'A5',6:'C1',7:'C2'})
        pdb_atoms['altLoc'] = ''
        # pdb_atoms['resName']    = self.atoms['molecule_name'].replace({'actin':'ACT','camkii':'CAM'})
        pdb_atoms['resName'] = self.atoms['resname']
        #pdb_atoms['chainID'] = self.atoms['molecule'].replace(cc_d)
        pdb_atoms['chainID'] = self.atoms['chainID']
#        assert False
        # pdb_atoms['resSeq']     = 0
        pdb_atoms['iCode'] = ''
        # pdb_atoms['x']          =
        # pdb_atoms['y']          =
        # pdb_atoms['z']          =
        pdb_atoms['occupancy'] = 0
        pdb_atoms['tempFactor'] = 0
        pdb_atoms['element'] = self.atoms['type'].replace(
            {1: 'C', 2: 'O', 3: 'N', 4: 'P', 5: 'H', 6: 'H', 7: 'H', 8: 'Mg', 9: 'Fe', 10: 'C'})
        pdb_atoms['charge'] = 0  # self.atoms['q'].astype(int)

        with open(file_name, 'w+') as f:
            chain = 'NoChain'
            resSeq = 0
            for i, a in pdb_atoms.iterrows():
                if a['chainID'] != chain:
                    resSeq = 1
                    chain = a['chainID']
                else:
                    resSeq += 1
                f.write(pdb_line % ('ATOM',
                                    int(a['serial']),
                                    a['name'].center(4),
                                    a['altLoc'],
                                    a['resName'],
                                    a['chainID'],
                                    a['resid'],
                                    a['iCode'],
                                    ('%8.3f' % (a['x'] / 10))[:8],
                                    ('%8.3f' % (a['y'] / 10))[:8],
                                    ('%8.3f' % (a['z'] / 10))[:8],
                                    a['occupancy'],
                                    a['tempFactor'],
                                    a['element'],
                                    a['charge']))
                                    
    def write_psf(self, file_name='actin.psf'):
        pass

    def write_gro(self, file_name='actin.gro', box_size=1000):
        gro_line = "%5d%-5s%5s%5d%8s%8s%8s%8s%8s%8s\n"
        pdb_atoms = self.atoms.copy()
        pdb_atoms['resName'] = self.atoms[
            'resname']  # self.atoms['molecule_name'].replace({'actin':'ACT','camkii':'CAM'})
        # pdb_atoms['name']       = self.atoms['type'].replace({1:'Aa',2:'Ab',3:'Ca',4:'Cb',5:'Da',6:'Db'})
        pdb_atoms['serial'] = np.arange(1, len(self.atoms) + 1)
        pdb_atoms['chainID'] = self.atoms['molecule']
        self.xmin, self.xmax = 0, box_size
        self.ymin, self.ymax = 0, box_size
        self.zmin, self.zmax = 0, box_size
        resSeq = 0

        with open(file_name, 'w+') as f:
            f.write('Generated Model\n')
            f.write('%5i\n' % len(pdb_atoms))
            chain = 'NoChain'
            resSeq = 0
            for i, a in pdb_atoms.iterrows():
                if a['molecule'] != chain:
                    resSeq = 1
                    chain = a['molecule']
                else:
                    resSeq += 1
                f.write(gro_line % (a['molecule'],
                                    a['resName'],
                                    a['name'],
                                    int(a['serial']),
                                    ('%8.3f' % (a['x'] / 10))[:8],
                                    ('%8.3f' % (a['y'] / 10))[:8],
                                    ('%8.3f' % (a['z'] / 10))[:8],
                                    '', '', ''))
            f.write(('   ' + ' '.join(['%8.3f'] * 3) + '\n') % (self.xmax, self.ymax, self.zmax))

    def print_coeff(self):
        if self.n_bonds > 0:
            self.bondtypes = self.bondtypes.sort_values('type')
            for i, b in self.bondtypes.iterrows():
                print('bond_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_angles > 0:
            for i, b in self.angletypes.iterrows():
                print('angle_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_dihedrals > 0:
            for i, b in self.dihedraltypes.iterrows():
                print('dihedral_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_impropers > 0:
            for i, b in self.impropertypes.iterrows():
                print('improper_coeff', int(b.type), b.k, '%.4f' % b['x0'])


class CoarseActin:

    @classmethod
    def from_parameters(cls,
                        box_size=10000,
                        n_actins=10,
                        n_camkiis=200,
                        min_dist=200,
                        align_actins=False,
                        bundle=False,
                        system2D=False,
                        model='Binding-Qian2',
                        sname='actin',
                        actinLenMin=50,
                        actinLenMax=100):
        self = cls()
        # Get actin coordinates actin
        pdb = prody.parsePDB(f'{__location__}/3j8i.pdb')
        mean = np.array([])
        for chain in 'DEF':
            selection = pdb.select('chain %s' % chain)
            D1 = pdb.select('chain %s and (resid 1 to 32 or resid 70 to 144 or resid 338 to 375)' % chain)
            D2 = pdb.select('chain %s and (resid 33 to 69)' % chain)
            D3 = pdb.select('chain %s and (resid 145 to 180 or resid 270 to 337)' % chain)
            D4 = pdb.select('chain %s and (resid 181 to 269)' % chain)
            m1 = D1.getCoords().mean(axis=0)
            m2 = D2.getCoords().mean(axis=0)
            m3 = D3.getCoords().mean(axis=0)
            m4 = D4.getCoords().mean(axis=0)
            mean = np.concatenate([mean, m1, m2, m3, m4], axis=0)
        mean = mean.reshape(-1, 3)
        actin = pandas.DataFrame(mean, columns=['x', 'y', 'z'])
        name = ['A1', 'A2', 'A3', 'A4'] * 3
        resid = [i for j in range(3) for i in [j] * 4]
        actin.index = zip(resid, name)
        # Build virtual sites
        vs = self.virtual_sites_definition
        for j in [2]:
            for i, s in vs[vs['molecule'] == model].iterrows():
                w12 = s['w12']
                w13 = s['w13']
                wcross = s['wcross']

                a = actin.loc[[(j, s['p1'])]].squeeze() / 10
                b = actin.loc[[(j, s['p2'])]].squeeze() / 10
                c = actin.loc[[(j, s['p3'])]].squeeze() / 10

                r12 = b - a
                r13 = c - a
                rcross = np.cross(r12, r13)
                r = (a + w12 * r12 + w13 * r13 + wcross * rcross) * 10
                r.name = (j, s['site'])
                actin = actin.append(r)
        actin_reference = actin.sort_index()
        # Build individual actins
        factin = []
        for i in range(n_actins):
            # Set actin length
            nactins = actinLenMin + int((actinLenMax-actinLenMin) * np.random.random())

            names = ['A1', 'A2', 'A3', 'A4'] * 2 + ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'] * (nactins - 2)
            resnames = ['ACD'] * (4 * 2) + ['ACT'] * (7 * (nactins - 2))
            resids = [1] * 4 + [2] * 4 + [i + 3 for j in range(nactins - 2) for i in [j] * 7]
            # actin_mass=41.74*1E3
            Factin = prody.AtomGroup()
            Factina = prody.AtomGroup()
            Factinb = prody.AtomGroup()

            Factin.setCoords(actin_reference)
            Factina.setCoords(actin_reference[4:-3])
            Factinb.setCoords(actin_reference[:-4 - 3])

            for i in range(nactins - 3):
                a, t = prody.superpose(Factina, Factinb)
                temp0 = Factin.getCoords()
                test = prody.applyTransformation(t, Factin)
                temp = np.concatenate([test.getCoords(), temp0[-4 - 3:]])
                # print(len(temp))
                Factin = prody.AtomGroup()
                Factina = prody.AtomGroup()
                Factinb = prody.AtomGroup()
                Factin.setCoords(temp)
                Factina.setCoords(temp[4:12])
                Factinb.setCoords(temp[0:8])

            Factin = prody.AtomGroup()
            Factin.setCoords(temp[:])
            n = len(Factin)
            Factin.setNames(names)
            Factin.setResnames(resnames)
            Factin.setResnums(resids)
            # Factin.setNames(['ALA' for i in range(n)])

            prody.writePDB('Factin.pdb', Factin)
            print(nactins, (n - 8) / 7. + 2)

            atoms = pandas.DataFrame(Factin.getCoords(), columns=['x', 'y', 'z'])
            atoms['q'] = -11
            atoms['molecule'] = 1
            atoms['type'] = [1, 2, 3, 4] * 2 + [1, 2, 3, 4, 5, 6, 7] * (nactins - 2)
            atoms['name'] = names
            # atoms['mass']=[D1_mass,D2_mass,D3_mass,D4_mass]*2+([D1_mass,D2_mass,D3_mass,D4_mass,0,0,0])*(nactins-2)
            atoms['resid'] = resids
            atoms['resname'] = resnames
            atoms.head()
            factin += [atoms.copy()]

        # Read camkii
        camkii = self.template
        # Build box
        actins = []
        camkiis = []
        for i in range(n_actins):
            d = 0
            while d < min_dist:
                f = factin[i][['x', 'y', 'z']].copy()
                f = f - f.mean()
                if align_actins:
                    rot = optimal_rotation(f)
                else:
                    rot = random_rotation()
                f = pandas.DataFrame(np.dot(rot, f[['x', 'y', 'z']].T).T, columns=f.columns)
                f = f - f.mean()
                f += [box_size / 2. for j in range(3)]
                a, b, c = [box_size * np.random.random() for j in range(3)]
                if bundle:
                    a = 0
                if system2D:
                    c = 0
                f += [a, b, c]
                f -= (f.mean() > box_size) * box_size
                f2 = factin[i].copy()
                f2[['x', 'y', 'z']] = f[['x', 'y', 'z']]
                # f+=[box_size/2. for i in range(3)]
                f2['molecule'] = i + 1
                f2['molecule_name'] = 'actin'
                f2['resname'] = factin[i]['resname']
                try:
                    d = sdist.cdist(f2[['x', 'y', 'z']], s[s['name'].isin(['A2', 'Cc'])][['x', 'y', 'z']]).min()
                except KeyError:
                    d = min_dist + 100
            actins += [f2]
            s = pandas.concat(actins)
        print("Actins in system")
        print(f"Total number of particles: {len(s)}")
        for i in range(n_camkiis):
            d = 0
            while d < min_dist:
                f = camkii[['x', 'y', 'z']].copy()
                f = f - f.mean()
                f = pandas.DataFrame(np.dot(random_rotation(), f[['x', 'y', 'z']].T).T, columns=f.columns)
                f = f - f.mean()
                f += [box_size / 2. for j in range(3)]
                a, b, c = [box_size * np.random.random() for j in range(3)]
                if system2D:
                    c = box_size/10 * np.random.random()
                f += [a, b, c]
                f -= (f.mean() > box_size) * box_size
                f2 = camkii.copy()
                f2[['x', 'y', 'z']] = f[['x', 'y', 'z']]
                # f+=[box_size/2. for i in range(3)]
                f2['molecule'] = n_actins + i + 1
                f2['molecule_name'] = 'camkii'
                f2['resid'] = i + 1
                f2['resname'] = 'CAM'
                # f2['mass']/=100
                # rr=np.random.randint(2)
                # if rr==1:
                # f2['type']+=2
                d = sdist.cdist(f2[['x', 'y', 'z']], s[s['name'].isin(['A2', 'Cc'])][['x', 'y', 'z']]).min()
            camkiis += [f2]
            s = pandas.concat(actins + camkiis,sort=True)
            print(f"CAMKII {i}") 
        print("CAMKIIs in system")
        print(f"Total number of particles: {len(s)}")
        s.index = np.arange(1, len(s) + 1)
        s['mass']=np.nan
        # Write system
        ss = SystemData(s.sort_values(['molecule', 'resid', 'name']))
        #ss.write_data(f'{sname}.data')
        ss.write_pdb(f'{sname}.pdb')
        ss.write_gro(f'{sname}.gro')
        ss.print_coeff()
        return self.from_topology(topology_file=f'{sname}.pdb', PlaneConstraint=system2D, periodic_box=box_size)

    @classmethod
    def from_topology(cls, topology_file='actin.pdb', periodic_box=10000, PlaneConstraint=False):
        self = cls()
        self.periodic_box = [periodic_box * 0.1] * 3
        self.forcefield = openmm.app.ForceField(f'{__location__}/ff.xml')
        self.top = openmm.app.PDBFile(topology_file)
        self.system = self.forcefield.createSystem(self.top.topology)
        self.system.setDefaultPeriodicBoxVectors(*np.diag(self.periodic_box))
        self.atom_list = self.parseTop()
        self.BuildVirtualSites()
        self.ComputeTopology()
        self.setForces(PlaneConstraint=PlaneConstraint)
        return self
        # Parse topology data

    def parseConfigurationFile(self, configuration_file=f'{__location__}/actinff.conf'):
        """Reads the configuration file for the forcefield"""
        self.configuration_file = configuration_file
        print(configuration_file)
        config = configparser.ConfigParser()
        config.read(configuration_file)
        self.template = parseConfigTable(config['Template'])
        self.bond_definition = parseConfigTable(config['Bonds'])
        self.bond_definition['type'] = '1' + self.bond_definition['i'].astype(str) + '-' + \
                                       (1 + self.bond_definition['s']).astype(str) + \
                                       self.bond_definition['j'].astype(str)
        self.angle_definition = parseConfigTable(config['Angles'])
        self.dihedral_definition = parseConfigTable(config['Dihedrals'])
        self.repulsion_definition = parseConfigTable(config['Repulsion'])
        self.virtual_sites_definition = parseConfigTable(config['Virtual sites'])

        self.bond_definition = self.bond_definition[self.bond_definition.molecule.isin(['Actin-ADP', 'CaMKII'])]
        self.angle_definition = self.angle_definition[self.angle_definition.molecule.isin(['Actin-ADP', 'CaMKII'])]
        self.dihedral_definition = self.dihedral_definition[
            self.dihedral_definition.molecule.isin(['Actin-ADP', 'CaMKII'])]
        # self.virtual_sites_definition = self.virtual_sites_definition[self.virtual_sites_definition.molecule.isin(['Actin-ADP', 'CaMKII','Binding-Qian2'])]

    def __init__(self):
        self.parseConfigurationFile()
        # self.forcefield = openmm.app.ForceField(f'{__location__}/ff.xml')
        # self.top = top
        # self.system = self.forcefield.createSystem(top.topology)
        # self.system.setDefaultPeriodicBoxVectors(*np.diag(periodic_box))
        # self.atom_list = self.parseTop()
        # self.BuildVirtualSites()
        # self.ComputeTopology()
        # self.setForces()
        # Parse topology data

    def parseTop(self):
        """ Converts the information from the topology to a table"""
        cols = ['atom_index', 'atom_id', 'atom_name',
                'residue_index', 'residue_id', 'residue_name',
                'chain_index', 'chain_id']
        data = []
        for residue in self.top.topology.residues():
            for atom in residue.atoms():
                data += [[atom.index, atom.id, atom.name,
                          residue.index, residue.id, residue.name,
                          residue.chain.index, residue.chain.id]]
        atom_list = pandas.DataFrame(data, columns=cols)
        atom_list.index = atom_list['atom_index']
        return atom_list

    def BuildVirtualSites(self):
        """ Sets the parameters for the virtual sites"""
        virtual_sites_definition = self.virtual_sites_definition.copy()
        virtual_sites_definition.index = [tuple(b) for a, b in
                                          virtual_sites_definition[['molecule', 'site']].iterrows()]

        # Actin binding sites parameters

        w1 = np.array(virtual_sites_definition.loc[[('Voth-Qian2020', 'A5')], ['w12', 'w13', 'wcross']].squeeze())
        w2 = np.array(virtual_sites_definition.loc[[('Voth-Qian2020', 'A6')], ['w12', 'w13', 'wcross']].squeeze())
        w3 = np.array(virtual_sites_definition.loc[[('Voth-Qian2020', 'A7')], ['w12', 'w13', 'wcross']].squeeze())

        # CAMKII virtual sites
        cw1 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C1')], ['w12', 'w13', 'wcross']].squeeze())
        cw2 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C2')], ['w12', 'w13', 'wcross']].squeeze())
        cw3 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C6')], ['w12', 'w13', 'wcross']].squeeze())
        cw4 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C7')], ['w12', 'w13', 'wcross']].squeeze())

        # Virtual sites
        for _, res in self.atom_list.groupby(['chain_index', 'residue_id']):
            assert len(res['residue_name'].unique()) == 1,print(len(res['residue_name'].unique()),_,res['residue_name'].unique())
            resname = res['residue_name'].unique()[0]
            ix = dict(list(zip(res['atom_name'], res['atom_index'])))
            if resname == 'ACT':
                # Virtual site positions
                a5 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w1[0], w1[1], w1[2])
                a6 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w2[0], w2[1], w2[2])
                a7 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w3[0], w3[1], w3[2])
                # Set up virtual sites
                self.system.setVirtualSite(ix['A5'], a5)
                self.system.setVirtualSite(ix['A6'], a6)
                self.system.setVirtualSite(ix['A7'], a7)
            if resname == 'CAM':
                # Parent sites
                c1 = ix['Cx1']
                c2 = ix['Cx2']
                c3 = ix['Cx3']
                # Virtual site positions
                c01 = openmm.OutOfPlaneSite(c1, c2, c3, cw1[0], cw1[1], cw1[2])
                c02 = openmm.OutOfPlaneSite(c1, c2, c3, cw2[0], cw2[1], cw2[2])
                c03 = openmm.OutOfPlaneSite(c2, c3, c1, cw1[0], cw1[1], cw1[2])
                c04 = openmm.OutOfPlaneSite(c2, c3, c1, cw2[0], cw2[1], cw2[2])
                c05 = openmm.OutOfPlaneSite(c3, c1, c2, cw1[0], cw1[1], cw1[2])
                c06 = openmm.OutOfPlaneSite(c3, c1, c2, cw2[0], cw2[1], cw2[2])
                c07 = openmm.OutOfPlaneSite(c1, c2, c3, cw3[0], cw3[1], cw3[2])
                c08 = openmm.OutOfPlaneSite(c1, c2, c3, cw4[0], cw4[1], cw4[2])
                c09 = openmm.OutOfPlaneSite(c2, c3, c1, cw3[0], cw3[1], cw3[2])
                c10 = openmm.OutOfPlaneSite(c2, c3, c1, cw4[0], cw4[1], cw4[2])
                c11 = openmm.OutOfPlaneSite(c3, c1, c2, cw3[0], cw3[1], cw3[2])
                c12 = openmm.OutOfPlaneSite(c3, c1, c2, cw4[0], cw4[1], cw4[2])
                cc = openmm.ThreeParticleAverageSite(c1, c2, c3, 1 / 3., 1 / 3., 1 / 3.)
                # Set up virtual positions
                self.system.setVirtualSite(ix['C01'], c01)
                self.system.setVirtualSite(ix['C02'], c02)
                self.system.setVirtualSite(ix['C03'], c03)
                self.system.setVirtualSite(ix['C04'], c04)
                self.system.setVirtualSite(ix['C05'], c05)
                self.system.setVirtualSite(ix['C06'], c06)
                self.system.setVirtualSite(ix['C07'], c07)
                self.system.setVirtualSite(ix['C08'], c08)
                self.system.setVirtualSite(ix['C09'], c09)
                self.system.setVirtualSite(ix['C10'], c10)
                self.system.setVirtualSite(ix['C11'], c11)
                self.system.setVirtualSite(ix['C12'], c12)
                self.system.setVirtualSite(ix['Cc'], cc)
        self.atom_list['Virtual'] = [self.system.isVirtualSite(a) for a in range(len(self.atom_list))]

    def ComputeTopology(self):
        # print(bonds)
        # Bonds, angles and dihedrals
        bonds = []
        angles = []
        dihedrals = []
        for _, c in self.atom_list.groupby('chain_index'):
            ix = {}
            for name, aa in c.groupby('atom_name'):
                ix.update({name: list(aa.index)})

            for SB, B in zip([bonds, angles, dihedrals],
                             [self.bond_definition, self.angle_definition, self.dihedral_definition]):
                for _, b in B.iterrows():
                    temp = pandas.DataFrame(columns=B.columns)
                    if 's' not in b:
                        b['s'] = 0

                    if b['i'] not in ix.keys():
                        continue

                    i1 = ix[b['i']][b['s']:]
                    i2 = ix[b['j']][:-b['s']] if b['s'] != 0 else ix[b['j']]
                    assert (len(i1) == len(i2))
                    temp['i'] = i1
                    temp['j'] = i2
                    if 'k' in b:
                        i3 = ix[b['k']]
                        assert (len(i1) == len(i3))
                        temp['k'] = i3
                    if 'l' in b:
                        i4 = ix[b['l']]
                        assert (len(i1) == len(i4))
                        temp['l'] = i4
                    for col in temp:
                        if col not in ['i', 'j', 'k', 'l']:
                            temp[col] = b[col]
                    SB += [temp]
        bonds = pandas.concat(bonds, sort=False)
        bonds.sort_values(['i', 'j'], inplace=True)
        angles = pandas.concat(angles, sort=False)
        angles.sort_values(['i', 'j', 'k'], inplace=True)
        dihedrals = pandas.concat(dihedrals, sort=False)
        dihedrals.sort_values(['i', 'j', 'k', 'l'], inplace=True)
        self.bonds = bonds.reset_index(drop=True)
        self.angles = angles.reset_index(drop=True)
        self.dihedrals = dihedrals.reset_index(drop=True)

    def Bond_diff(self, coord):
        # Comparison to starting structure distances
        import scipy.spatial.distance as sdist
        real_dist = sdist.squareform(sdist.pdist(coord.getPositions(asNumpy=True)))
        for i, b in self.bonds.iterrows():
            self.bonds.at[i, 'xr'] = real_dist[b['i'], b['j']] * 10
        self.bonds['diff'] = ((self.bonds['xr'] - self.bonds['r0']) ** 2) ** .5
        return self.bonds.groupby('type').mean().sort_values('diff', ascending=False)

    def clearForces(self):
        """ Removes all forces from the system """
        [self.system.removeForce(0) for i, f in enumerate(self.system.getForces())]

    def setForces(self, PlaneConstraint=False, CaMKII_Force='multigaussian', BundleConstraint=False):
        """ Adds the forces to the system """
        self.clearForces()
        # Harmonic Bonds
        harmonic_bond = openmm.HarmonicBondForce()
        harmonic_bond.setUsesPeriodicBoundaryConditions(True)
        for i, b in self.bonds.iterrows():
            harmonic_bond.addBond(int(b['i']), int(b['j']), b['r0'] / 10., b['K'] * 4.184 * 100)
        self.system.addForce(harmonic_bond)

        # Harmonic angles
        harmonic_angle = openmm.HarmonicAngleForce()
        harmonic_angle.setUsesPeriodicBoundaryConditions(True)
        for i, b in self.angles.iterrows():
            harmonic_angle.addAngle(int(b['i']), int(b['j']), int(b['k']), b['t0'] / 180 * np.pi, b['K'] * 4.184)
        self.system.addForce(harmonic_angle)

        # Harmonic torsions
        harmonic_torsion = openmm.PeriodicTorsionForce()
        harmonic_torsion.setUsesPeriodicBoundaryConditions(True)
        for i, b in self.dihedrals.iterrows():
            harmonic_torsion.addTorsion(int(b['i']), int(b['j']), int(b['k']), int(b['l']), b['period'],
                                        b['t0'] / 180 * np.pi, b['K'] * 4.184)
        self.system.addForce(harmonic_torsion)

        # Repulsion
        for i, r in self.repulsion_definition.iterrows():
            # print(r)
            # print(r['force'].format(i))
            rf = openmm.CustomNonbondedForce('(epsilon{0}*((sigma{0}/r)^12-2*(sigma{0}/r)^6)+epsilon{0})*step(sigma{0}-r)'.format(i))
            rf.setNonbondedMethod(rf.CutoffPeriodic)
            rf.addGlobalParameter('epsilon{0}'.format(i), r['epsilon'])
            rf.addGlobalParameter('sigma{0}'.format(i), r['sigma'])
            rf.setCutoffDistance(10)
            rf.setUseLongRangeCorrection(False)
            for _, a in self.atom_list.iterrows():
                rf.addParticle()
            sel1 = self.atom_list[self.atom_list['atom_name'] == r['i']]
            sel2 = self.atom_list[self.atom_list['atom_name'] == r['j']]
            rf.addInteractionGroup(sel1.index, sel2.index)
            rf.createExclusionsFromBonds(self.bonds[['i', 'j']].values.tolist(), 2)
            self.system.addForce(rf)

        # Donors
        Cm = [self.atom_list[self.atom_list['atom_name'] == 'C01'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C02'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C03'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C04'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C05'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C06'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C07'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C08'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C09'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C10'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C11'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C12'].index]
        Cc = self.atom_list[self.atom_list['atom_name'] == 'Cc'].index
        comb = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]

        if CaMKII_Force=='multigaussian':
            for i, j in comb:
                gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                         "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                         "dd=(dist1^2+dist2^2+dist3^2)/3;"
                                                         "dist1= distance(a1,d1);"
                                                         "dist2= min(distance(a2,d2),distance(a2,d3));"
                                                         "dist3= min(distance(a3,d2),distance(a3,d3));")

                gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
                gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
                gaussian.addGlobalParameter('w1', 5.0)  # well1 width
                gaussian.addGlobalParameter('w2', 0.5)  # well2 width
                gaussian.setCutoffDistance(12)

                # Aceptors
                A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
                A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
                A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
                assert len(A1) == len(A2) == len(A3)
                for a1, a2, a3 in zip(A1, A2, A3):
                    gaussian.addAcceptor(a1, a2, a3)

                # Donors
                for d1, d2, d3 in zip(Cc, Cm[i], Cm[j]):
                    gaussian.addDonor(d1, d2, d3)

                self.system.addForce(gaussian)
        elif CaMKII_Force=='doublegaussian':
            for i, j in comb:
                gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                         "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                         "dd=(dist2^2+dist3^2)/2;"
                                                         "dist2= min(distance(a2,d2),distance(a2,d3));"
                                                         "dist3= min(distance(a3,d2),distance(a3,d3));")

                gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
                gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
                gaussian.addGlobalParameter('w1', 5.0)  # well1 width
                gaussian.addGlobalParameter('w2', 0.5)  # well2 width
                gaussian.setCutoffDistance(12)

                # Aceptors
                A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
                A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
                A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
                assert len(A1) == len(A2) == len(A3)
                for a1, a2, a3 in zip(A1, A2, A3):
                    gaussian.addAcceptor(a1, a2, a3)

                # Donors
                for d1, d2, d3 in zip(Cc, Cm[i], Cm[j]):
                    gaussian.addDonor(d1, d2, d3)

                self.system.addForce(gaussian)
        if CaMKII_Force=='singlegaussian':
            gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                     "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                     "dd= distance(a1,d1);")

            gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
            gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
            gaussian.addGlobalParameter('w1', 5.0)  # well1 width
            gaussian.addGlobalParameter('w2', 0.5)  # well2 width
            gaussian.setCutoffDistance(12)

            # Aceptors
            A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
            A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
            A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
            assert len(A1) == len(A2) == len(A3)
            for a1, a2, a3 in zip(A1, A2, A3):
                gaussian.addAcceptor(a1, -1, -1)

            # Donors
            for d1 in Cc:
                gaussian.addDonor(d1, -1, -1)

            self.system.addForce(gaussian)
        


        if PlaneConstraint:
            print(self.periodic_box)
            midz = self.periodic_box[-1] / 2 / 10
            print(midz)
            plane_constraint = openmm.CustomExternalForce('kp*(z-mid)^2')
            plane_constraint.addGlobalParameter('mid', midz)
            plane_constraint.addGlobalParameter('kp', 0.001)
            for i in self.atom_list.index:
                plane_constraint.addParticle(i, [])
            self.system.addForce(plane_constraint)
        else:
            plane_constraint = openmm.CustomExternalForce('kp*0')
            plane_constraint.addGlobalParameter('kp', 0.001)
            for i in self.atom_list.index:
                plane_constraint.addParticle(i, [])
            self.system.addForce(plane_constraint)

        if BundleConstraint:
            print('Bundle Constraint added')
            bundle_constraint = openmm.CustomCentroidBondForce(2, 'kp_bundle*(distance(g1,g2)^2-(x1-x2)^2)')
            bundle_constraint.addGlobalParameter('kp_bundle',0.01)
            bundle_constraint.setUsesPeriodicBoundaryConditions(True)
            cc = 0
            for c, chain in self.atom_list.groupby('chain_index'):
                if 'ACT' in chain.residue_name.unique():
                    print(f'Setting up Bundle constraint for chain c')
                    bundle_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[:16]))
                    print(len(list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[:16])))
                    bundle_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:]))
                    print(len(list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:])))
                    print([cc,cc+1])
                    bundle_constraint.addBond([cc, cc + 1])
                    cc += 2
            self.system.addForce(bundle_constraint)
            print(self.system.getNumForces())
        else:
            print('Bundled constrain not added')
            bundle_constraint = openmm.CustomCentroidBondForce(2, '0*kp_bundle*(distance(g1,g2)^2-(x1-x2)^2)')
            bundle_constraint.addGlobalParameter('kp_bundle',0.01)
            bundle_constraint.setUsesPeriodicBoundaryConditions(True)
            cc = 0
            for c, chain in self.atom_list.groupby('chain_index'):
                if 'ACT' in chain.residue_name.unique():
                    bundle_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[16:]))
                    bundle_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:]))
                    bundle_constraint.addBond([cc, cc + 1])
                    cc += 2
            self.system.addForce(bundle_constraint)
            print(self.system.getNumForces())
        '''
	if BundleConstraint:
            bundle_constraint = openmm.CustomCompoundBondForce(2,'kb*((y1-y2)^2+(z1+z2)^2')
            bundle_constraint.addGlobalParameter('kb', 0.1)
            for i in self.atom_list.index:
                bundle_constraint.addBond(i, [])
            self.system.addForce(bundle_constraint)
        else:
            bundle_constraint = openmm.CustomExternalForce('kb*0')
            bundle_constraint.addGlobalParameter('kb', 0.1)
            for i in self.atom_list.index:
                bundle_constraint.addParticle(i, [])
            self.system.addForce(bundle_constraint)
        '''


# Tests

def test_basic_MD():
    # Create a system

    # Test that there is no error
    pass


# Actin flexibility
def test_actin_persistence_length():
    '''Determines the persistence length of a filament in a simulation.'''
    # Simulate a big actin

    # Measure the persistence length
    
class HexGrid():
    deltas = [[1,0,-1],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0]]
    a0=0
    a1=np.pi/3
    a2=-np.pi/3
    vecs=np.array([[np.sqrt(3)*np.cos(a0),np.sin(a0)/np.sqrt(3)],
                   [np.sqrt(3)*np.cos(a1),np.sin(a1)/np.sqrt(3)],
                   [np.sqrt(3)*np.cos(a2),np.sin(a2)/np.sqrt(3)]])
    def __init__(self, radius):
        self.radius = radius
        self.tiles = {(0, 0, 0): "X"}
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexas_in_edge = r
                for i in range(num_of_hexas_in_edge):
                    a = a+self.deltas[j][0]
                    b = b+self.deltas[j][1]
                    c = c+self.deltas[j][2]           
                    self.tiles[a,b,c] = "X"
                    
    def coords(self):
        tiles=np.array([a for a in hg.tiles.keys()])
        coords=np.dot(tiles,self.vecs)
        return coords

    def show(self):
        l = []
        for y in range(20):
            l.append([])
            for x in range(60):
                l[y].append(".")
        for (a,b,c), tile in self.tiles.items():
            l[self.radius-1-b][a-c+(2*(self.radius-1))] = self.tiles[a,b,c]
        mapString = ""
        for y in range(len(l)):
            for x in range(len(l[y])):
                mapString += l[y][x]
            mapString += "\n"
        print(mapString)

if __name__=='__main__':
    print(__name__)
    ###################################
    #Setting Conditions for simulation#
    ###################################

    parameters={"epsilon":[100],
                "aligned":[False],
                "actinLen":[500],
                "layers":[2],
    #            "repetition":range(3),
                "disorder":[.5,.75],
                "temperature":[300],
                "system2D":[False],
                "frequency":[1000],
                "run_time":[20],
                "CaMKII_Force":['multigaussian','doublegaussian','singlegaussian'],
                "simulation_platform":["OpenCL"]}
    test_parameters={"simulation_platform":"CUDA",
                    "frequency":1000,
                    "run_time":1,
                    "CaMKII_Force":'doublegaussian'
                    }
    job_id=0
    if len(sys.argv)>1:
        try:
            job_id=int(sys.argv[1])
        except TypeError:
            pass
    sjob=SlurmJobArray("ActinBundle", parameters, test_parameters,job_id)
    sjob.print_parameters()
    sjob.print_slurm_variables()
    sjob.write_csv()

    print ("name :", sjob.name)


    ##############
    # Parameters #
    ##############
    aligned=sjob["aligned"]
    system2D=sjob["system2D"]
    actinLen=sjob["actinLen"]
    Sname=sjob.name
    simulation_platform=sjob["simulation_platform"]


    ###################
    # Build the model #
    ###################
    #Set the points in the actin network 
    import string
    import random
    bound_actin_template=pandas.read_csv("CaMKII_bound_with_actin.csv",index_col=0)
    def add_bound_actin(full_model, length=100,
                        twist=2.89942054, shift=-28.21600347,
                        rotation=np.array([[1.,0.,0.],
                                           [0.,1.,0.],
                                           [0.,0.,1.]]),
                        translation=np.array([5000,5000,5000])):

        q = np.array([[np.cos(twist), -np.sin(twist), 0, 0],
                      [np.sin(twist), np.cos(twist), 0, 0],
                      [0, 0, 1, shift],
                      [0, 0, 0, 1]])
        rot = q[:3, :3].T
        trans = q[:3, 3]

        #Create the points
        point=bound_actin_template[['x','y','z']]
        points = []
        for i in range(length):
            points += [point]
            point = np.dot(point, rot) + trans
        points = np.concatenate(points)

        #Create the model
        model = pandas.DataFrame(points, columns=['x', 'y', 'z'])
        model["resid"] = [j+i for i in range(length) for j in bound_actin_template["resid"]]
        model["name"] = [j for i in range(length) for j in bound_actin_template["name"]]
        model["type"] = [j for i in range(length) for j in bound_actin_template["type"]]
        model["resname"]=[j for i in range(length) for j in bound_actin_template["resname"]]


        #Remove two binding points
        model=model[~((model['resid']>length-1) & (model['name'].isin(['A5','A6','A7']+['Cc']+[f'C{i+1:02}' for i in range(12)]+[f'Cx{i+1}' for i in range(3)])))]

        #Remove all CaMKII except resid 50
        #model=model[~((model['resid']!=50) & (model['resname'].isin(['CAM'])))]

        model.loc[model[model['resid']==model['resid'].max()].index,'resname']='ACD'
        model.loc[model[model['resid']==model['resid'].min()].index,'resname']='ACD'
        for chain_name in string.ascii_uppercase+string.ascii_lowercase:
            #print(chain_name)
            if chain_name in full_model['chainID'].values:
                model.loc[model['resname'].isin(['ACT','ACD']),'chainID']=chain_name
                continue
            model.loc[model['resname'].isin(['ACT','ACD']),'chainID']=chain_name
            break
            
        for chain_name in string.ascii_uppercase+string.ascii_lowercase:
            #print(chain_name,'A' in model['chainID'])
            if chain_name in full_model['chainID'].values or chain_name in model['chainID'].values:
                model.loc[model['resname'].isin(['CAM']),'chainID']=chain_name
                continue
            model.loc[model['resname'].isin(['CAM']),'chainID']=chain_name
            break

        #model["name"] = [j for i in range(1000) for j in ['A1', 'A2', 'A3', 'A4']]

        #Center the model
        model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

        #Move the model
        model[['x', 'y', 'z']]=np.dot(model[['x', 'y', 'z']], rotation) + translation
        
        full_model=pandas.concat([full_model,model])
        full_model.index=range(len(full_model))
        return full_model

    full_model=pandas.DataFrame(columns=['chainID'])

    if sjob["layers"]==1:
        hg=HexGrid(2)
        coords=hg.coords()[:2]
        d=59.499*2    
    else:
        hg=HexGrid(sjob["layers"])
        coords=hg.coords()
        d=59.499*2

    for c in coords:
            height=(random.random()-0.5)*39*28.21600347*sjob["disorder"]
            print(c[0],c[1],height)
            full_model=add_bound_actin(full_model, length=sjob["actinLen"], translation=np.array([5000+d*c[0],5000+d*c[1],5000+height]))

    #Remove the CaMKII that are not overlapping
    sel=full_model[full_model['name']=='Cc']
    i=sel.index
    d=sdist.pdist(sel[['x','y','z']])
    d=pandas.Series(d,itertools.combinations(i,2))
    sel2=sel.loc[[a for a,b in d[d<35].index]]
    print(len(sel2))
    full_model.loc[:,'chain_resid']=full_model[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1)
    print(len(full_model[full_model['resname'].isin(['ACT','ACD'])]))
    print(len(full_model[full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]))

    full_model=full_model[full_model['resname'].isin(['ACT','ACD']) | 
                          full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]
    print(len(full_model))

    #Remove the CaMKII that are colliding
    sel=full_model[full_model['name']=='Cc']
    i=sel.index
    d=sdist.pdist(sel[['x','y','z']])
    d=pandas.Series(d,itertools.combinations(i,2))
    sel2=sel.loc[[b for a,b in d[d<35].index]]
    print(len(sel2))
    full_model.loc[:,'chain_resid']=full_model[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1)
    print(len(full_model[full_model['resname'].isin(['ACT','ACD'])]))
    print(len(full_model[full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]))

    full_model=full_model[~full_model['chain_resid'].isin(sel2[['chainID','resid',]].apply(lambda x:''.join([str(a) for a in x]),axis=1))]

    full_model['mass'] = 1
    full_model['molecule'] = 1
    full_model['q'] = 0
    ss = SystemData(full_model.sort_values(['chainID', 'resid', 'name']))
    ss.write_data()
    ss.write_pdb(f'{Sname}.pdb')
    ss.write_gro(f'{Sname}.gro')
    ss.print_coeff()

    ##############
    # Simulation #
    ##############
    import sys
    sys.path.insert(0,'.')
    import openmm
    import openmm.app
    from simtk.unit import *
    import time
    from sys import stdout

    time.ctime()
    platform = openmm.Platform.getPlatformByName(simulation_platform)

    #Create system
    s=CoarseActin.from_topology(f'{Sname}.pdb',)
    print("System initialized")
    s.setForces(BundleConstraint=aligned,PlaneConstraint=system2D,
                CaMKII_Force=sjob['CaMKII_Force'])
    top=openmm.app.PDBFile(f'{Sname}.pdb')
    coord=openmm.app.GromacsGroFile(f'{Sname}.gro')


    #Set up simulation
    temperature=sjob["temperature"]*kelvin
    integrator = openmm.LangevinIntegrator(temperature, .0001/picosecond, 1*picoseconds)
    simulation = openmm.app.Simulation(top.topology, s.system, integrator,platform)
    simulation.context.setPositions(coord.positions)


    #Modify parameters
    simulation.context.setParameter("g_eps", sjob["epsilon"])

    frequency=sjob["frequency"]
    #Add reporters
    simulation.reporters.append(openmm.app.DCDReporter(f'{Sname}.dcd', frequency),)
    simulation.reporters.append(openmm.app.StateDataReporter(stdout, frequency, step=True,time=True,potentialEnergy=True, temperature=True,separator='\t',))
    simulation.reporters.append(openmm.app.StateDataReporter(f'{Sname}.log', frequency, step=True,time=True,totalEnergy=True, kineticEnergy=True,potentialEnergy=True, temperature=True))

    #Print initial energy
    state = simulation.context.getState(getEnergy=True)
    energy=state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print (f'Initial energy: {energy} KJ/mol')

    #Run
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature*kelvin)
    time0=time.ctime()
    time_0=time.time()
    #simulation.step(100000)

    #Turn off nematic parameter
    #simulation.context.setParameter('kp_bundle',0)
    simulation.runForClockTime(sjob["run_time"])

    #Save checkpoint
    chk=f'{Sname}.chk'
    simulation.saveCheckpoint(chk)

    #simulation.step(100000000)
