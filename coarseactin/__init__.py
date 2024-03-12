"""
CoarseActinModel
Coarse Grained Model of Actin
"""

# Add imports here
#from .system import *
#from .actin import *
from .Scene import *
from .System import *
from .components import *
from .utils.SlurmJobArray import SlurmJobArray
from .utils.HexGrid import HexGrid
from .utils.ChainNameGenerator import chain_name_generator
from .mdfit import MDFit

# Handle versions
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


