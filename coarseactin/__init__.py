"""
CoarseActinModel
Coarse Grained Model of Actin
"""

# Add imports here
#from .system import *
#from .actin import *
from .Scene import *
from .components import *
from .utils.SlurmJobArray import SlurmJobArray
from .utils.HexGrid import HexGrid
# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


