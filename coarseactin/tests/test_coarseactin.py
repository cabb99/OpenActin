"""
Unit and regression test for the coarseactin package.
"""

# Import package, test suite, and other packages as needed
import coarseactin
import pytest
import sys

def test_coarseactin_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "coarseactin" in sys.modules
