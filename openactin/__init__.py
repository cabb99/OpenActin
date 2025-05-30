"""A coarse-grained model of actin filaments based on Voth 4-particle model"""

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
from ._version import __version__


