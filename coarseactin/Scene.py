import pandas
import numpy

'''
Python library to allow easy handling of coordinate files for molecular dynamics using pandas DataFrames.
'''


if __name__ == "__main__":
    import utils
else:
    from . import utils


class Scene(pandas.DataFrame):

    # Required
    protein_residues = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    # Initialization
    def __init__(self, particles, **kwargs):
        """Create an empty scene"""
        pandas.DataFrame.__init__(self, particles)
        self.__dict__['_meta'] = {}
        assert 'x' in self.columns
        assert 'y' in self.columns
        assert 'z' in self.columns
        if 'chain_index' not in self.columns:
            self['chain_index'] = 1
        if 'res_index' not in self.columns:
            self['res_index'] = 1
        if 'name' not in self.columns:
            self['name'] = [f'P{i:03}' for i in range(len(self))]
        for attr, value in kwargs.items():
            self._meta[attr] = value

    @classmethod
    def from_pdb(cls, file):
        def pdb_line(line):
            l = dict(recname=str(line[0:6]).strip(),
                     serial=int(line[6:11]),
                     name=str(line[12:16]).strip(),
                     altLoc=str(line[16:17]),
                     resname=str(line[17:20]).strip(),
                     chainID=str(line[21:22]),
                     resSeq=int(line[22:26]),
                     iCode=str(line[26:27]),
                     x=float(line[30:38]),
                     y=float(line[38:46]),
                     z=float(line[46:54]),
                     occupancy=line[54:60].strip(),
                     tempFactor=line[60:66].strip(),
                     element=str(line[76:78]),
                     charge=str(line[78:80]))
            if l['occupancy'] == '':
                l['occupancy'] = 1.0
            else:
                l['occupancy'] = float(l['occupancy'])
            if l['tempFactor'] == '':
                l['tempFactor'] = 1.0
            else:
                l['tempFactor'] = float(l['tempFactor'])
            return l

        with open(file, 'r') as pdb:
            lines = []
            mod_lines=[]
            for i, line in enumerate(pdb):
                if len(line) > 6:
                    header = line[:6]
                    if header == 'ATOM  ' or header == 'HETATM':
                        lines += [pdb_line(line)]
                    elif len(line) > 6 and header == "MODRES":
                        m = dict(recname=str(line[0:6]).strip(),
                                 idCode=str(line[7:11]),
                                 resname=str(line[12:15]).strip(),
                                 chainID=str(line[16:17]),
                                 resSeq=int(line[18:22]),
                                 iCode=str(line[22:23]),
                                 stdRes=str(line[24:27]).strip(),
                                 comment=str(line[29:70]))
                        mod_lines += [m]
        pdb_atoms = pandas.DataFrame(lines)
        pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                               'resname', 'chainID', 'resSeq', 'iCode',
                               'x', 'y', 'z', 'occupancy', 'tempFactor',
                               'element', 'charge']]

        kwargs={}
        if len(mod_lines) > 0:
            kwargs.update(dict(modified_residues=pandas.DataFrame(mod_lines)))

        chain_map = {b: a for a, b in enumerate(pdb_atoms['chainID'].unique())}
        pdb_atoms['chain_index'] = pdb_atoms['chainID'].replace(chain_map)
        pdb_atoms['res_index'] = pdb_atoms['resname']

        return cls(pdb_atoms,**kwargs)

    @classmethod
    def from_cif(cls, file):
        data = []
        with open(file) as f:
            reader = utils.PdbxReader(f)
            reader.read(data)
        block = data[0]
        atom_data = block.getObj('atom_site')
        cif_atoms = pandas.DataFrame([atom_data.getFullRow(i) for i in range(atom_data.getRowCount())],
                                     columns=atom_data.getAttributeList(),
                                     index=range(atom_data.getRowCount()))
        cif_atoms[['x', 'y', 'z']] = cif_atoms[['Cartn_x', 'Cartn_y', 'Cartn_z']]
        return cls(cif_atoms)

    @classmethod
    def from_gro(cls, gro):
        return cls()

    # Writing
    def write_pdb(self, file):
        # Fill empty columns
        pdb_table = self.copy()
        pdb_table['serial'] = numpy.arange(1, len(self) + 1) if 'serial' not in pdb_table else pdb_table['serial']
        pdb_table['name'] = 'A' if 'name' not in pdb_table else pdb_table['name']
        pdb_table['altLoc'] = '' if 'altLoc' not in pdb_table else pdb_table['altLoc']
        pdb_table['resName'] = 'R' if 'resName' not in pdb_table else pdb_table['resName']
        pdb_table['chainID'] = 'C' if 'chainID' not in pdb_table else pdb_table['chainID']
        pdb_table['resSeq'] = 1 if 'resSeq' not in pdb_table else pdb_table['resSeq']
        pdb_table['iCode'] = '' if 'iCode' not in pdb_table else pdb_table['iCode']
        assert 'x' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdb_table.columns, 'Coordinate x not in particle definition'
        pdb_table['occupancy'] = 0 if 'occupancy' not in pdb_table else pdb_table['occupancy']
        pdb_table['tempFactor'] = 0 if 'tempFactor' not in pdb_table else pdb_table['tempFactor']
        pdb_table['element'] = '' if 'element' not in pdb_table else pdb_table['element']
        pdb_table['charge'] = 0 if 'charge' not in pdb_table else pdb_table['charge']

        # Override chain names if molecule is present
        if 'molecule' in pdb_table:
            import string
            cc = (string.ascii_uppercase.replace('X', '') + string.ascii_lowercase + '1234567890' + 'X') * 1000
            cc_d = dict(zip(range(1, len(cc) + 1), cc))
            pdb_table['chainID'] = self.atoms['molecule'].replace(cc_d)

        # Write pdb file
        with open(file, 'w+') as pdb:
            for i, atom in pdb_table.iterrows():
                line = f'ATOM  {i:>5} {atom["name"]:^4} {atom["resName"]:<3} {atom["chainID"]}{atom["resSeq"]:>4}' + \
                       '    ' + \
                       f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' + ' ' * 22 + f'{atom.element:2}' + ' ' * 2
                assert len(line) == 80, 'An item in the atom table is longer than expected'
                pdb.write(line + '\n')

    def write_cif(self, file):
        """Write a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        entry : str=None
            The entry ID to assign to the CIF file
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        """Write out a model to a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        modelIndex : int=1
            The model number of this frame
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        # Fill empty columns
        pdbx_table = self.copy()
        pdbx_table['serial'] = numpy.arange(1, len(self) + 1) if 'serial' not in pdbx_table else pdbx_table['serial']
        pdbx_table['name'] = 'A' if 'name' not in pdbx_table else pdbx_table['name']
        pdbx_table['altLoc'] = '?' if 'altLoc' not in pdbx_table else pdbx_table['altLoc']
        pdbx_table['resName'] = 'R' if 'resName' not in pdbx_table else pdbx_table['resName']
        pdbx_table['chainID'] = 'C' if 'chainID' not in pdbx_table else pdbx_table['chainID']
        pdbx_table['resSeq'] = 1 if 'resSeq' not in pdbx_table else pdbx_table['resSeq']
        pdbx_table['resIC'] = 1 if 'resIC' not in pdbx_table else pdbx_table['resIC']
        pdbx_table['iCode'] = '' if 'iCode' not in pdbx_table else pdbx_table['iCode']
        assert 'x' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdbx_table.columns, 'Coordinate x not in particle definition'
        pdbx_table['occupancy'] = 0 if 'occupancy' not in pdbx_table else pdbx_table['occupancy']
        pdbx_table['tempFactor'] = 0 if 'tempFactor' not in pdbx_table else pdbx_table['tempFactor']
        pdbx_table['element'] = 'C' if 'element' not in pdbx_table else pdbx_table['element']
        pdbx_table['charge'] = 0 if 'charge' not in pdbx_table else pdbx_table['charge']
        pdbx_table['model'] = 0 if 'model' not in pdbx_table else pdbx_table['model']

        with open(file, 'w+') as pdbx:
            pdbx.write('data_pdbx\n')
            pdbx.write('#\n')
            pdbx.write('loop_\n')
            pdbx.write('_atom_site.group_PDB\n')
            pdbx.write('_atom_site.id\n')
            pdbx.write('_atom_site.type_symbol\n')
            pdbx.write('_atom_site.label_atom_id\n')
            pdbx.write('_atom_site.label_alt_id\n')
            pdbx.write('_atom_site.label_comp_id\n')
            pdbx.write('_atom_site.label_asym_id\n')
            pdbx.write('_atom_site.label_entity_id\n')
            pdbx.write('_atom_site.label_seq_id\n')
            pdbx.write('_atom_site.pdbx_PDB_ins_code\n')
            pdbx.write('_atom_site.Cartn_x\n')
            pdbx.write('_atom_site.Cartn_y\n')
            pdbx.write('_atom_site.Cartn_z\n')
            pdbx.write('_atom_site.occupancy\n')
            pdbx.write('_atom_site.B_iso_or_equiv\n')
            pdbx.write('_atom_site.Cartn_x_esd\n')
            pdbx.write('_atom_site.Cartn_y_esd\n')
            pdbx.write('_atom_site.Cartn_z_esd\n')
            pdbx.write('_atom_site.occupancy_esd\n')
            pdbx.write('_atom_site.B_iso_or_equiv_esd\n')
            pdbx.write('_atom_site.pdbx_formal_charge\n')
            pdbx.write('_atom_site.auth_seq_id\n')
            pdbx.write('_atom_site.auth_comp_id\n')
            pdbx.write('_atom_site.auth_asym_id\n')
            pdbx.write('_atom_site.auth_atom_id\n')
            pdbx.write('_atom_site.pdbx_PDB_model_num\n')
            for i, atom in pdbx_table.iterrows():
                line = f"ATOM  {i:>5} {atom['element']:^3} {atom['name']:^4} . {atom['resName']:^4} " + \
                       f"{atom['chainID']} ? {atom['resSeq']:^5} {atom['resIC']} " + \
                       f"{atom['x']:10.4f} {atom['y']:10.4f} {atom['z']:10.4f}  0.0  0.0  ?  ?  ?  ?  ?  .  " + \
                       f"{atom['resSeq']:5} {atom['resName']:4} {atom['chainID']} {atom['name']:^4} {atom['model']:^5}"
                pdbx.write(line + '\n')
            pdbx.write('#\n')

    def write_gro(self, file):
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

        with open(file, 'w+') as f:
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

    #get methods
    def get_coordinates(self):
        return self[['x', 'y', 'z']]

    def get_sequence(self):
        pass

    def set_coordinates(self, coordinates):
        self[['x', 'y', 'z']] = coordinates

    def copy(self, deep=True):
        return Scene(super().copy(deep), **self._meta)

    def correct_modified_aminoacids(self):
        out = self.copy()
        for i, row in out.modified_residues.iterrows():
            sel = ((out['resname'] == row['resname']) &
                   (out['chainID'] == row['chainID']) &
                   (out['resSeq'] == row['resSeq']))
            out.loc[sel, 'resname'] = row['stdRes']
        return out

    # Built ins
    def __repr__(self):
        return f'<Scene ({len(self)} particles)>\n{super().__repr__()}'

    def __add__(self, other):
        if isinstance(other, self.__class__):
            concatenated = pandas.concat([self, other], axis=0)
            return Scene(concatenated)
        else:
            coord = self.get_coordinates()
            temp = self.copy()
            temp.set_coordinates(coord+other)
            return temp

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            return other+s
        else:
            return Scene(super().__add__(other))

    def __mul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(coord*other)
        return temp

    def __rmul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(other*coord)
        return temp

    def __getattr__(self, attr):

        if attr in self._meta:
            return self._meta[attr]
        else:
            raise AttributeError(f"type object {str(self.__class__)} has no attribute {str(attr)}")

    def __setattr__(self, attr, value):
        self._meta[attr] = value

    """
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        elif 'particles' in self.__dict__:
            return getattr(self.particles, attr)
        else:
            if '__repr__' in self.__dict__:
                raise AttributeError(f"type object {str(self)} has no attribute {str(attr)}")
            else:
                raise AttributeError()

    def __setattr__(self, attr, value):
        if 'particles' in self.__dict__ and attr in self.particles.__dict__:
            return setattr(self.particles, attr, value)
        else:
            self.__dict__[attr] = value
    """

if __name__ == '__main__':
    particles = pandas.DataFrame([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 columns=['x', 'y', 'z'])
    s1 = Scene(particles)
    s1.write_pdb('test.pdb')
    s2 = Scene.from_pdb('test.pdb')

    s2.write_cif('test.cif')
    s3 = Scene.from_cif('test.cif')
    s3.write_pdb('test2.pdb')
    s4 = Scene.from_pdb('test.pdb')


    s1.to_csv('particles_1.csv')
    s2.to_csv('particles_2.csv')
    s3.to_csv('particles_3.csv')
    s4.to_csv('particles_4.csv')

"""
import numpy as np
import pandas as pd

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

a = pd.DataFrame(
    data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))

filename = '/tmp/data.h5'
metadata = dict(local_tz='US/Eastern')
h5store(filename, a, **metadata)
with pd.HDFStore(filename) as store:
    data, metadata = h5load(store)

print(data)
#     A   B   C   E   D
# 0   9  20  92  43  25
# 1   2  64  54   0  63
# 2  22  42   3  83  81
# 3   3  71  17  64  53
# 4  52  10  41  22  43
# 5  48  85  96  72  88
# 6  10  47   2  10  78
# 7  30  80   3  59  16
# 8  13  52  98  79  65
# 9   6  93  55  40   3
"""
