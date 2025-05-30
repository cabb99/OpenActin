"""
Unit and regression test for the openactin package.
"""

# Import package, test suite, and other packages as needed
import openactin
import pytest
import sys

def test_openactin_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openactin" in sys.modules

def test_create_actin_system():
    pass

def test_simulate_actin_system():
    pass

def test_actin_diffussion():
    """Checks that the translational and rotational diffussion are as expected"""

    assert True  # Check the translational diffussion
    assert True  # Check the rotational diffussion
    pass

def test_actin_stretching():
    """Checks that the strectching constant is as expected"""
    pass

def test_actin_bending():
    """
    Checks that the bending constant is as expected

    """
    pass

def test_actin_twisting_clockwise():
    pass

def test_actin_twisting_anticlockwise():
    pass

def test_actin_persistence_length():
    """Check that the persistence length is as expected"""
    "17uM+-2"

def test_actin_bonds():
    pass

"""Check that the actin binding sites don't move too much from their expected position during the simulation"""
