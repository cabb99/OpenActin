import nose
import unittest
import coarseactin


def test_Scene_exists():
    coarseactin.Scene

def test_from_matrix():
    s = coarseactin.Scene([[0, 0, 0],
                           [0, 0, 1]])
    assert len(s) == 2

def test_from_numpy():
    import numpy as np
    a = np.random.random([100,3])*100
    s = coarseactin.Scene(a)
    assert len(s) == 100

def test_from_dataframe():
    import numpy as np
    import pandas
    a = np.random.random([100,3])*100
    atoms = pandas.DataFrame(a,columns=['z','y','x'])
    s = coarseactin.Scene(atoms)
    assert s['x'][20] == atoms['x'][20]



class Test_Read_Write():
    def _convert(self, reader, writer,mol):
        if reader == 'pdb':
            s1 = coarseactin.Scene.from_pdb(f'data/{mol}.pdb')
        elif reader == 'cif':
            s1 = coarseactin.Scene.from_cif(f'data/{mol}.cif')
        elif reader == 'gro':
            s1 = coarseactin.Scene.from_gro(f'data/{mol}.gro')
        elif reader == 'fixPDB_pdb':
            s1 = coarseactin.Scene.from_fixPDB(pdbfile=f'data/{mol}.pdb')
        elif reader == 'fixPDB_cif':
            s1 = coarseactin.Scene.from_fixPDB(pdbxfile=f'data/{mol}.cif')
        elif reader == 'fixPDB_pdbid':
            s1 = coarseactin.Scene.from_fixPDB(pdbid=f'{mol}')

        if writer == 'pdb':
            fname = f'scratch/{reader}_{writer}_{mol}.pdb'
            s1.write_pdb(fname)
            s2 = coarseactin.Scene.from_pdb(fname)
        elif writer == 'cif':
            fname = f'scratch/{reader}_{writer}_{mol}.cif'
            s1.write_cif(fname)
            s2 = coarseactin.Scene.from_cif(fname)
        elif writer == 'gro':
            fname = f'scratch/{reader}_{writer}_{mol}.gro'
            s1.write_gro(fname)
            s2 = coarseactin.Scene.from_gro(fname)

        s1.to_csv('scratch/s1.csv')
        s2.to_csv('scratch/s2.csv')
        print(len(s1))
        assert(len(s1) == len(s2)), f"The number of particles before reading ({len(s1)}) and after writing ({len(s2)})"\
                                    f" are different.\nCheck the file: {fname}"


    def test_convert(self):
        for reader in ['pdb','cif','gro']: # ,'fixPDB_pdb','fixPDB_cif','fixPDB_pdbid']:
            for writer in ['pdb','cif','gro']:
                for mol in ['1r70', '1zbl', '1zir']:
                    yield self._convert, reader, writer, mol


    def test_pdb2pdb(self):
        for mol in ['1r70', '1zbl', '1zir', '3wu2']:
            yield self._convert, 'pdb', 'pdb', mol

    #def _cif_to_cif(self, mol):
    #    s1 = coarseactin.Scene.from_cif(f'data/{mol}')
    #    s1.write_cif(f'scratch/{mol}')
    #    s2 = coarseactin.Scene.from_cif(f'data/{mol}')
    #    assert len(s1) == len(s2)

    #def test_cif(self):
    #    for mol in ['1r70', '1zbl', '1zir', '3wu2']:
    #        yield self._cif_to_cif, f'{mol}.cif'



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

#def test_write_read_cif():
#    for mol in ['1r70.cif', '1zbl.cif', '1zir.cif', '3wu2.cif', '4v99.cif']:
#        s1 = coarseactin.Scene.from_cif(f'data/{mol}')
#        s1.write_cif(f'scratch/{mol}')
#        s2 = coarseactin.Scene.from_cif(f'data/{mol}')
#        assert len(s1) == len(s2)



def test_write_read_pdb2():
    assert True

def test_write_read_pdb3():
    assert True

def setup_func():
    "set up test fixtures"
    a=True

def teardown_func():
    "tear down test fixtures"
    a=False

@nose.with_setup(setup_func, teardown_func)
def test():
    "test ..."
    assert True



def test_evens():
    for i in range(0, 5):
        yield check_even, i*2, i*3

def check_even(n, nn):
    assert n % 2 == 0 or nn % 2 == 0

def setup_module(module):
    print ("") # this is to get a newline after the dots
    print ("setup_module before anything in this file")

def teardown_module(module):
    print ("teardown_module after everything in this file")

def my_setup_function():
    print ("my_setup_function")

def my_teardown_function():
    print ("my_teardown_function")

def multiply(a,b):
    return a*b

@nose.with_setup(my_setup_function, my_teardown_function)
def test_numbers_3_4():
    print ('test_numbers_3_4  <============================ actual test code')
    assert multiply(3,4) == 12

@nose.with_setup(my_setup_function, my_teardown_function)
def test_strings_a_3():
    print ('test_strings_a_3  <============================ actual test code')
    assert multiply('a',3) == 'aaa'


class TestUM:

    def setup(self):
        print ("TestUM:setup() before each test method")

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_numbers_5_6(self):
        print ('test_numbers_5_6()  <============================ actual test code')
        assert multiply(5,6) == 30

    def test_strings_b_2(self):
        print ('test_strings_b_2())  <============================ actual test code')
        assert multiply('b',2) == 'bb'


if __name__ == '__main__':
    pass
