import pytest
from coarseactin import SlurmJobArray
import numpy as np


@pytest.fixture
def sjob1():
    parameters = {"var1": np.linspace(1, 5, 10),
                  "var2": np.logspace(0, 1, 5),
                  "var3": ['x', 'y', 'z'],
                  "constant1": [False],
                  "constant2": [100],
                  "constant3": ['Temperature'],
                  }
    test_parameters = {"var2": 1,
                       "var3": 'y',
                       "constant2": 200,
                       "constant3": 'Pressure',
                       }
    return SlurmJobArray("Folder_name/S", parameters, test_parameters, 'test')


@pytest.fixture
def sjob2():
    parameters = {"var1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  "var2": range(100),
                  "var3": ['x', 'y', 'z'],
                  "constant1": [False],
                  "constant2": [100],
                  "constant3": ['Temperature'],
                  }
    test_parameters = {"var2": 1,
                       "var3": 'y',
                       "constant2": 200,
                       "constant3": 'Pressure',
                       }
    return SlurmJobArray("Folder_name/S2", parameters, test_parameters, 1)


def test_len(sjob1, sjob2):
    assert len(sjob1) == 150
    assert len(sjob2) == 3000


def test_get_parameter(sjob1):
    assert sjob1.test_run
    assert sjob1['constant2'] == 200
    sjob1.test_run = False
    assert not sjob1.test_run
    assert sjob1['constant2'] == 100
    with pytest.raises(KeyError):
        c = sjob1['constant4']


def test_wrong_test_parameter():
    sjob_good = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, {'y': 0}, job_id=3)
    with pytest.raises(KeyError):
        sjob_bad = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, {'z': 0}, job_id=3)


def test_simple_job():
    sjob = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, job_id=3)
    assert sjob.name == 'S_003_x_1_y_1'


def test_test_job():
    sjob = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, {'y': 2}, 'test')
    assert sjob.name == 'S_test_x_0_y_2'


def test_none_job():
    sjob = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, {'y': 2}, None)
    assert sjob.name == 'S_000_x_0_y_0'


def test_small_job():
    sjob = SlurmJobArray('S', {'x': [0, 1], 'y': [0, 1]}, {'y': 2}, 3)
    assert sjob.name == 'S_003_x_1_y_1'


def test_large_job():
    sjob = SlurmJobArray('S', {'x': range(100), 'y': range(11)}, {'y': 2}, 5)
    assert sjob.name == 'S_0005_x_0_y_5'


def test_write_jobs():
    sjob = SlurmJobArray('S', {'x': range(20), 'y': range(20)}, {'y': 2}, 5)
    last_job = sjob.write_jobs().split('\n')[-2].strip()
    s = "python /home/cb/Development/CoarseGrainedActin/coarseactin/utils/SlurmJobArray.py 399 > S_399_x_19_y_19.log"

    assert last_job.split(' ')[-1] == 'S_399_x_19_y_19.log'
    assert last_job.split(' ')[-3] == '399'
    assert last_job.split(' ')[-4] == __file__
