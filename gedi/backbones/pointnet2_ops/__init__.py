from . import pointnet2_modules
from . import pointnet2_utils

_jit_radius_ext = getattr(pointnet2_utils, "gedi_radius_search_op", None)
if _jit_radius_ext is not None:
	gedi_radius_search_op = _jit_radius_ext

try:
	from . import gedi_radius_search_op as gedi_radius_search_op
except ImportError:
	if _jit_radius_ext is None:
		raise
