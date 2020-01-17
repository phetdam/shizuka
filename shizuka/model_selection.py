# implements some useful methods for model selection. intended as a complement
# to the more general methods available in sklearn or other libraries.
#
# Changelog:
#
# 01-16-2020
#
# initial creation. copied from resampled_cv.py from tomodachi_proj project, and
# then rebranded. started work on the resampled_cv function; added error checks.
#
__doc__ = """
contains methods for more specialized model selection.

not intended as a replacement for a more complete machine learning package like
sklearn, but as a complement for the more general model selection routines.
"""

_MODULE_NAME = "shizuka.model_selection"

from numpy import ravel
from pandas import DataFrame
from imblearn.base import SamplerMixin
from inspect import getfullargspec
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold
from sys import stderr

## internal constants
# accepted scoring methods for resampled_cv
_scorings = ["r2", "accuracy"]


def resampled_cv(est, X_train, y_train, scoring = None, resampler = None,
                 resampler_kwargs = None, cv = 3, n_jobs = 1):
    """
    evaluates supervised learning models with cross-validation and optional or
    built-in resampling. avoids the naive error of validating with a fold of
    rsampled training data; the original training data used in training folds
    is copied and validated on the untouched validation fold.

    requires the package imblearn, which can be found on PyPI.

    parameters:

    est               sklearn regression or classification estimator, a subclass
                      of ClassifierMixin or RegressorMixin from sklearn.base
    X_train           2d training feature matrix, must be ndarray or DataFrame.
    y_train           1d training response vector, same number of observations
                      as X_train, ex. ndarray, Series, or one-column DataFrame.
    scoring     optional metric used to compute score of the estimator, default
                None which indicates that the estimator's builtin score function
                is to be used. right now the function only supports "accuracy"
                for classifiers and "r2" for regressors.
    resampler   optional resampling object or function to use for resampling, or
                a recognized resampling method such as "SMOTE" or "SMOTEENN".
                acceptable resampling objects must inherit from SamplerMixin
                defined in imblearn.base, i.e. they must implement fit_resample,
                whose call signature is fit_resample(self, X, y), where X has
                shape (n_samples, n_features) and y has shape (n_samples,),
                returning Xr, yr with shape (n_samples_new, n_features),
                (n_samples). a user-defined resampling function must have the
                same call signature as fit_resample, without the self arg.
    resampler_kwargs  keyword arguments to be passed to the SamplerMixin or
                      resampling function defined by the resampler arg.
    cv                optional number of CV folds, default 3. must be > 2.
    n_jobs            number of jobs to run in parallel. multiprocessing will
                      be used for the function's main code but implementation of
                      multiprogramming is resampler dependent.
    """
    # save function name
    fn = resampled_cv.__name__
    ## sanity check boilerplate
    # check that the estimator est is a classifier. if not, print error
    if isinstance(est, ClassifierMixin) == False:
        raise TypeError("{0}: est must be classifier inheriting from "
                        "sklearn.base.ClassifierMixin".format(_fn))
    # check that X_train and y_train have same length
    if len(X_train) != len(y_train):
        raise ValueError("{0}: X_train and y_train must have same number of "
                         "observations".format(_fn))
    # check type of X_train and y_train
    if not isinstance(X_train, DataFrame):
        raise TypeError("{0}: X_train should be a pandas DataFrame"
                        "".format(_fn))
    # be more careful when checking type of y_train
    if hasattr(y_train, "__iter__") and (not isinstance(y_train, str)):
        # if y_train is a DataFrame, check that there is only one column
        if isinstance(y_train, DataFrame):
            if len(y_train.columns) > 1:
                raise ValueError("{0}: y_train ({1}) must have one column only"
                                 "".format(_fn, type(y_train)))
        # else if y_train is a numpy array, also check ndim
        elif isinstance(y_train, ndarray):
            if y_train.ndim != 1:
                raise ValueError("{0}: y_train ({1}) must have ndim == 1"
                                 "".format(_fn, type(y_train)))
        # don't allow dictionary
        elif isinstance(y_train, dict):
            raise TypeError("{0}: y_train ({1}) must be a 1d iterable"
                            "".format(_fn))
        # don't allow multidimensionality
        for val in y_train:
            if (not isinstance(val, str)) and hasattr(val, "__iter__"):
                raise TypeError("{0}: y_train must be a 1d iterable"
                                "".format(_fn))
    else:
        raise TypeError("{0}: y_train should be a pandas DataFrame with one "
                        "column or a 1d iterable (ndarray, etc.)"
                        "".format(_fn))
    # check scoring metric (only accept None or "r2", "accuracy")
    if scoring is None: pass
    elif isinstance(scoring, str):
        if scoring not in _scorings:
            raise ValueError("{0}: unsupported scoring metric \"{1}\". only "
                             "supported metrics are {2}"
                             "".format(_fn, scoring, _scorings))
    else: raise TypeError("{0}: scoring must be None or a string".format(_fn))
    # check resampler/resampling function (can be None); check if SamplerMixin
    if (resampler is None) or isinstance(resampler, SamplerMixin): pass
    # else if a function (more accurately, is callable), check call signature
    elif hasattr(resampler, "__call__"):
        # get full arg spec and check call signature
        fas = getfullargspec(resampler)
        # check that there are only two positional args; i.e. length of args -
        # number of defaults is 2 (args includes named args), and does not
        # allow any more positional args. if not, error
        if (len(fas.args) - len(fas.defaults) == 2) and (fas.varargs == None):
            pass
        else:
            raise ValueError("{0}: resampling function f must have signature "
                             "with only 2 positional args, >= 0 named args, "
                             "and optional **kwargs".format(_fn))
    else:
        raise TypeError("{0}: resampler type must be imblearn.base.SamplerMixin"
                        "or function".format(_fn))
    # check that resampler_kwargs is dict or None
    if (resampler_kwargs is None) or isinstance(resampler_kwargs, dict): pass
    else:
        raise TypeError("{0}: resampler_kwargs must be None or dict"
                        "".format(_fn))
    # check cv folds
    if isinstance(cv, int):
        if cv >= 3: pass
        else: raise ValueError("{0}: int cv must be an >= 3".format(_fn))
    else: raise TypeError("{0}: cv must be an int >= 3")
    # check n_jobs
    if isinstance(n_jobs, int):
        # sometimes -1 is passed to specify using all cores
        if (n_jobs > 0) or (n_jobs == -1): pass
        else:
            raise ValueError("{0}: int n_jobs must be positive or -1"
                             "".format(_fn))
    else: raise TypeError("{0}: n_jobs must be a positive int or -1"
                          "".format(_fn))
    # use KFold
        

# main
if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
