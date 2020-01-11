# contains some utility functions that may be shared by several modules.
#
# Changelog:
#
# 01-10-2020
#
# initial creation. migrated _get_cmap_colors from plotting.py.

_MODULE_NAME = "shizuka._utils"

__doc__ = ""

from matplotlib import cm
from matplotlib.colors import ListedColormap
from sys import stderr

def _get_cmap_colors(cmap, n, clim = 1, callfn = None):
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
    clim      optional float, default 1, where 0 < clim <= 1. clim controls the
              overall contrast of the returned n colors; lower values of clim
              result in colors that are less contrasting, for fixed n.
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
    if (not isinstance(clim, float)) and (not isinstance(clim, int)):
        raise TypeError("{0}: error: clim must be a float in range (0, 1]"
                        "".format(callfn))
    if (clim <= 0) or (clim > 1):
        raise ValueError("{0}: error: float clim outside range (0, 1]"
                         "".format(callfn))
    # take range [0, clim] and split it into n pieces; the collected points are
    # endpoints of each interval. then, choose a point from the middle of each
    # interval; this reduces the contrast of the chosen colors.
    ends = [clim * (i + 1) / n for i in range(n)]
    colors = [None for _ in range(n)]
    colors[0] = ends[0] / 2
    for i in range(1, n): colors[i] = (ends[i] + ends[i - 1]) / 2
    # try to get the colormap
    try: cmap = getattr(cm, cmap)
    except AttributeError as ae:
        ae.args = ["{0}: error: unknown color map \"{1}\"".format(callfn)]
    # retrieve colors using color points and return
    for i in range(n): colors[i] = cmap(colors[i])
    # return ListedColormap from colors
    return ListedColormap(colors, name = cmap.name + "_listed", N = n)

# main
if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
