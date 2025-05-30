"""
Unit and regression test for the openactin package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openactin


def test_openactin_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openactin" in sys.modules
