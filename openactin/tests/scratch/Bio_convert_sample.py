from Bio.PDB import PDBParser, MMCIFIO
from Bio.PDB import PDBIO, MMCIFParser

test_structures=['1r70', '1zbl', '1zir', '3wu2']

for structure in test_structures:
    p = PDBParser()
    struc = p.get_structure("", f"../data/{structure}.pdb")
    io = MMCIFIO()
    io.set_structure(struc)
    io.save(f"pdb2cif_{structure}.cif")
    
for structure in test_structures:
    p = PDBParser()
    struc = p.get_structure("", f"../data/{structure}.pdb")
    io = PDBIO()
    io.set_structure(struc)
    io.save(f"pdb2pdb{structure}.pdb")
    
for structure in test_structures:
    p = MMCIFParser()
    struc = p.get_structure("", f"../data/{structure}.cif")
    io = MMCIFIO()
    io.set_structure(struc)
    io.save(f"cif2cif_{structure}.cif")
    
for structure in test_structures:
    p = MMCIFParser()
    struc = p.get_structure("", f"../data/{structure}.cif")
    io = PDBIO()
    io.set_structure(struc)
    io.save(f"cif2pdb{structure}.pdb")
