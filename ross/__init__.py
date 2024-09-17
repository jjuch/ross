__version__ = "1.5.0"
from plotly import io as _pio

import ross.plotly_theme

from .bearing_seal_element import *
from .faults import *
from .disk_element import *
from .materials import *
from .point_mass import *
from .probe import *
from .results import *
from .rotor_assembly import *
from .multi_rotor import *
from .shaft_element import *
from .units import Q_
from .utils import get_data_from_figure, visualize_matrix
from ross.fluid_flow.lubricants import lubricant_dict

_pio.templates.default = "ross"
