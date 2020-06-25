# __init__.py
#
# Changelog:
#
# 06-24-2020
#
# modified __all__ entry, updated docstring, and added simple dependency check.
# split dependency check into required and optional dependencies. used warnings
# module instead of printing to sys.stderr.
#
# 01-10-2020
#
# initial creation. add __doc__ for package level, __all__, and version.

from importlib import import_module
from warnings import warn

__doc__ = """Simplifying the analysis of statistical model performance.

Making easier the analysis of statistical model performance by providing an
abstracting layer over the standard Python scientific computing stack.

Requires ``sklearn >=0.22``.

.. note::

   Dependency list subject to change since package is in active development.
"""
__all__ = ["model_selection", "plotting", "utils"]
__version__ = "0.0.1"

# check required dependencies
_req_deps = ["matplotlib", "numpy", "pandas", "sklearn"]
for _d in _req_deps:
    try: import_module(_d)
    except ImportError:
        warn("warning: missing required dependency {0}".format(_d),
              category = ImportWarning)

# check optional dependencies
_opt_deps = ["sphinx-rtd-theme"]
for _d in _opt_deps:
    try: import_module(_d)
    except ImportError:
        warn("warning: missing optional dependency {0}".format(_d),
             category = ImportWarning)
