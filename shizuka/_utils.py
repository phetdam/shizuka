# contains some utility functions that may be shared by several modules.
#
# Changelog:
#
# 01-12-2020
#
# corrected unraised AttributeError, and added additional check to make sure
# retrieved attribute from matplotlib.cm is a Colormap, not something else.
# adjusted the effect of _get_cmap_colors so that when the cc (formerly clim)
# parameter is increased from 0 to 1, the colors from the interval [0, 1] are
# increasingly clustered around the middle, 0.5.
#
# 01-10-2020
#
# initial creation. migrated _get_cmap_colors from plotting.py.

_MODULE_NAME = "shizuka._utils"

__doc__ = ""

from matplotlib import cm
from matplotlib.colors import Colormap, ListedColormap
from sys import stderr

def _get_cmap_colors(cmap, n, cc = 0, callfn = None):
    """
    internal function that takes a string describing a known matplotlib color
    map, and returns n colors from that color map as a ListedColormap. the
    string itself not refer to a ListedColormap; LinearSegmentedColormap names
    are fine as well, which unlocks the entire matplotlib color map suite. the
    colors returned are equidistant in contrast relative to the specified map.

    note: the matplotlib.cm.get_cmap function, by default, tries to maximize the
          color contrast. _get_cmap_colors does not do this; it tries to choose
          colors in a way that the contrast is moderated.

    parameters:

    cmap      string for a matplotlib ListedColormap or LinearSegmentedColormap
    n         int number of colors to return
    cc        optional float, default 0, where 0 <= cc < 1. cc controls the
              overall contrast of the returned n colors; higher values of cc
              result in colors that are less contrasting, for fixed n, that are
              increasingly restricted to an interval symmetric around the middle
              of the color gradient defined by the selected color map. in other
              words, given a standardized color mapping range [0, 1], cc != 0
              gives the interval [0.5 * cc, 1 - 0.5 * cc].
    callfn    optional string name of the calling function, default None
    """
    if callfn is None: callfn = _get_cmap_colors.__name__
    # sanity checks
    if not isinstance(cmap, str):
        raise TypeError("{0}: error: cmap must be a string matplotlib color "
                        "map".format(callfn))
    if not isinstance(n, int):
        raise TypeError("{0}: error: n must be a positive integer"
                        "".format(callfn))
    if n < 1:
        raise ValueError("{0}: error: int n must be positive".format(callfn))
    if (not isinstance(cc, float)) and (not isinstance(cc, int)):
        raise TypeError("{0}: error: cc must be a float in range [0, 1)"
                        "".format(callfn))
    if (cc < 0) or (cc >= 1):
        raise ValueError("{0}: error: float cc outside range [0, 1)"
                         "".format(callfn))
    # take range [0.5 * cc, 1 - 0.5 * cc] and split it into n pieces; the
    # collected points are midpoints of each interval. reduces color contrast.
    colors = [0.5 * cc + (1 - cc) * (i + 0.5) / n for i in range(n)]
    # try to get the colormap
    try: cmap = getattr(cm, cmap)
    except AttributeError as ae:
        ae.args = ["{0}: error: unknown color map \"{1}\"".format(callfn, cmap)]
        raise ae
    # if cmap is not a Colormap, raise a TypeError
    if not isinstance(cmap, Colormap):
        raise TypeError("{0}: error: {1} is not a valid Colormap"
                        "".format(callfn, cmap))
    # retrieve colors using color points and return
    for i in range(n): colors[i] = cmap(colors[i])
    # return ListedColormap from colors
    return ListedColormap(colors, name = cmap.name + "_listed", N = n)

# main
if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
