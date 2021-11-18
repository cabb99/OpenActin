"""
CoarseActinModel
Coarse Grained Model of Actin
"""

# Add imports here
#from .system import *
#from .actin import *
from .Scene import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions


