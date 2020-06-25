# contains various utility functions.
#
# Changelog:
#
# 06-24-2020
#
# initial creation. added functions for checking if an estimator is compatible
# with the shizuka API or if the estimator is quasi-scikit-learn compatible.
# added checks to is_compat and is_sklearn_compat to ensure that the attributes
# being tested for are instance methods, not functions, with function is_method.

from inspect import getfullargspec
from sys import stderr

__doc__ = """Contains utility functions used throughout the package."""

def is_method(obj):
    """Determines if an object is an bound method or not.

    :param obj: Object to evaluate
    :type obj: object
    :returns: ``True`` if ``obj`` is an instance method, ``False`` otherwise
    :rtype: bool
    """
    return str(type(obj)) == "method"

def is_compat(est):
    """Determines if an estimator is compatible with the ``shizuka`` API.

    The exact notion of "compatibility" is covered in :doc:`../model_compat`.
    Please read that page for details.

    .. note::

       If ``est`` is ``None``, then ``False`` will be silently returned.

    :param est: Concrete estimator class instance.
    :type est: object
    :returns: ``True`` if ``est`` is compatible, ``False`` otherwise.
    :rtype: bool
    """
    # compatible if and only if not None and has a get_params instance method
    # note that we explicitly check that get_params is a method, not a function
    if (est is not None) and hasattr(est, "get_params") and \
       hasattr(est.get_params, "__call__") and is_method(est.get_params):
        # get_params must take exactly 0 positional arguments
        _argspec = getfullargspec(est.get_params)
        if len(_argspec.args) == len(_argspec.defaults): return True
    return False

def is_sklearn_compat(est):
    """Determines if an estimator is quasi-scikit-learn compatible.

    The exact notion of "quasi-scikit-learn compatibility", which is a rather
    loose definition, is covered in :doc:`../model_compat`. Please
    read that page for details.

    .. note::

       Calls :func:`is_compat` internally, so will return ``False`` if ``est``
       is ``None``.

    :param est: Concrete estimator class instance.
    :type est: object
    :returns: ``True`` if ``est`` is quasi-scikit-learn compatible, ``False``
        otherwise.
    :rtype: bool
    """
    # if not compatible, return False
    if is_compat(est) == False: return False
    # need to get all 3 boolean flags to be True in order to return True
    fit_ok, predict_ok, score_ok = False, False, False
    # check if fit method is implemented, with correct call signature
    if hasattr(est, "fit") and hasattr(est.fit, "__call__") and \
       is_method(est.fit):
        _argspec = getfullargspec(est.fit)
        # must have exactly 2 positional args
        if (len(_argspec.args) >= 2) and \
           (len(_argspec.args) - len(_argspec.defaults) == 2):
            fit_ok = True
    # check if predict method is implemented, with correct call signature
    if hasattr(est, "predict") and hasattr(est.predict, "__call__") and \
       is_method(est.predict):
        _argspec = getfullargspec(est.predict)
        # must have exactly 1 positional arg
        if (len(_argspec.args) >= 1) and \
           (len(_argspec.args) - len(_argspec.defaults) == 1):
            predict_ok = True
    # check if score method is implemented, with correct call signature
    if hasattr(est, "score") and hasattr(est.score, "__call__") and \
       is_method(est.score):
        _argspec = getfullargspec(est.score)
        # must have exactly 2 positional args
        if (len(_argspec.args) >= 2) and \
           (len(_argspec.args) - len(_argspec.defaults) == 2):
            score_ok = True
    # return True if and only if fit, predict, and score are all ok
    if (fit_ok and predict_ok and score_ok): return True
    return False

if __name__ == "__main__":
    print("{0}: do not run module as script.".format(__module__), file = stderr)
