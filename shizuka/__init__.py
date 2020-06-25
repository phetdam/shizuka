# __init__.py
#
# Changelog:
#
# 06-24-2020
#
# modified __all__ entry, updated docstring, and added simple dependency check.
#
# 01-10-2020
#
# initial creation. add __doc__ for package level, __all__, and version.

from sys import stderr

__doc__ = """Simplifying the analysis of statistical model performance.

Making easier the analysis of statistical model performance by providing an
abstracting layer over the standard Python scientific computing stack.

Requires ``sklearn >=0.22``.

.. note::

   Dependency list subject to change since package is in active development.
"""
__all__ = ["model_selection", "plotting", "utils"]
__version__ = "0.0.1"

# check dependencies
_deps = ["matplotlib", "numpy", "pandas", "sklearn"]
for _d in _deps:
    try: import _d
    except ImportError:
        print("WARNING: missing required dependency {0}".format(_d),
              file = stderr)
