# implements some useful methods for model selection. intended as a complement
# to the more general methods available in sklearn or other libraries.
#
# Changelog:
#
# 01-17-2020
#
# added dict of internally supported samplers (so you don't have to pass a
# SamplerMixin or function to resampled_cv), and worked on docstring format to
# indicate what the returned results will look like. changed type checking to
# consider if resampler class is abc.ABCMeta, and updated docstring. untested.
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

from abc import ABCMeta
from numpy import ravel
from pandas import DataFrame
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from inspect import getfullargspec
from numpy import mean, std
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sys import stderr
from time import time

## internal constants
# accepted scoring methods for resampled_cv
_scorings = ["r2", "accuracy"]
# supported internally specified resampling methods for resampled_cv
_samplers = {"SMOTE": SMOTE, "ADASYN": ADASYN, "SMOTEENN": SMOTEENN}

def resampled_cv(est, X_train, y_train, scoring = None, resampler = None,
                 resampler_kwargs = None, shuffle = False, random_state = None,
                 cv = 3, n_jobs = 1):
    """
    evaluates supervised learning models with cross-validation and optional or
    built-in resampling. avoids the naive error of validating with a fold of
    rsampled training data; the original training data used in training folds
    is copied and validated on the untouched validation fold.

    returns a dict of results given by

    {"best_estimator": a, "best_cv_score": b, "train_scores": [...],
     "cv_scores": [...], "mean_cv_score": c, "std_cv_score": d, 
     "train_times": [...], "total_time": e, "resampler": None or f, 
     "resampler_kwargs": None or {...}, "resampling_times": None or [...],
     "train_shapes": [...], "cv_shapes": [...], "cv_iter": g, 
     "shuffle": True or False, "random_state": None or h}

    "best_estimator" gives the fitted estimator object that had the best CV
    score, while "best_cv_score" gives that best estimator's CV score. 
    "train_scores" gives the training scores of the estimator for each CV
    iteration, while "cv_scores" gives validation scores of the estimator for
    each CV iteration. "mean_cv_score" is the average validation score across
    iterations, "std_cv_score" is the standard deviation of validation scores,
    "train_times" indicates the time in seconds it took the estimator to be
    trained on the [resampled] training data for each CV iteration. "total_time"
    indicates the total overall running time in seconds, "resampler" is a
    string, function, or abc.ABCMeta indicating the resampling callable, and 
    "resampler_kwargs" may be None or a dict of keyword args passed to the
    resampler. "resampling_times" gives the runtimes of calling the resampler on
    the training data for each CV iteration, "train_shapes" and "cv_shapes" give
    the shape of the [resampled] training and validation data subsets for each
    CV iteration. "cv_iter" gives the number of CV iterations (folds). "shuffle"
    and "random_state" correspond to the keyword arguments passed to the
    sklearn.model_selection.KFold object instantiated within the method for
    generating the cross-validation data set splits. see parameters below.

    requires the package imblearn, which can be found on PyPI.

    note: standard deviation of CV scores is computed with denominator cv - 1.

    parameters:

    est               sklearn regression or classification estimator, a subclass
                      of ClassifierMixin or RegressorMixin from sklearn.base.
                      should be unfitted and initialized with hyperparameters.
    X_train           2d training feature matrix, required to be DataFrame
    y_train           1d training response vector, same number of observations
                      as X_train, ex. ndarray, Series, or one-column DataFrame.
    scoring           optional metric used to compute score of the estimator, 
                      default None which indicates that the estimator's builtin 
                      score function is to be used. function only supports 
                      "accuracy" for classifiers and "r2" for regressors.
    resampler         optional resampling class name or function for resampling,
                      or a recognized resampling method such as "SMOTE" or 
                      "SMOTEENN". acceptable resampling classes must inherit 
                      from abc.ABCMeta and implement fit_resample, whose call 
                      signature is given by fit_resample(self, X, y). X has 
                      shape (n_samples, n_features). y has shape (n_samples,),
                      fit_resample returns Xr, yr with shape (n_samples_new, 
                      n_features), (n_samples). a user-defined resampling 
                      function must have two unnamed required parameters X, y
                      with optional keyword arguments or named arguments.

                      note: if passing a user-defined resampling function, that
                            function must copy its input data internally!
                       
    resampler_kwargs  dict of keyword arguments to be passed to the abc.ABCMeta
                      upon subclass instantiation or resampling function
    shuffle           optional boolean, default False, that indicates whether or
                      not to shuffle indices before splitting.
    random_state      optional, default False. sets the seed for the random
                      number generator; ignored if shuffle = False.
    cv                optional number of CV folds, default 3. must be > 2.
    n_jobs            number of jobs to run in parallel. multiprocessing will
                      be used for the function's main code but implementation of
                      multiprogramming is resampler dependent.

                      note: currently unsupported; single thread only.
    """
    # get function start time
    time_start = time()
    # save function name
    fn = resampled_cv.__name__
    ## sanity check boilerplate
    # check that the estimator est is a regressor or classifier. if not, error
    if (isinstance(est, ClassifierMixin) == False) and \
       (isinstance(est, RegressorMixin) == False):
        raise TypeError("{0}: est must be a classifier or regressor inheriting "
                        "from sklearn.base.ClassifierMixin or sklearn.base."
                        "RegressorMixin".format(_fn))
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
    # check resampler/resampling function (can be None); check if resampler is
    # abc meta class, implementing the fit_resample method
    if (resampler is None) or isinstance(resampler, ABCMeta):
        if hasattr(resampler, "fit_resample"): pass
        else:
            raise NotImplementedError("{0}: {1} has no fit_resample method"
                                      "".format(_fn, resampler))
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
    # else if it is one of the internally supported sampling methods
    elif isinstance(resampler, str) and (resampler in _samplers): pass
    else:
        raise TypeError("{0}: resampler type must be abc.ABCMeta implementing "
                        "fit_transform, a function, or supported resampling "
                        "method in {1}".format(_fn, tuple(_samplers.keys())))
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
    # create results dict
    cv_results = {}
    # set some easy attributes, like number of cv iterations
    cv_results["cv_iter"] = cv
    cv_results["shuffle"] = shuffle
    cv_restuls["random_state"] = random_state
    # may be None, function, or abc.ABCMeta
    cv_results["resampler"] = resampler
    # None or dict
    cv_results["resampler_kwargs"] = resampler_kwargs
    # set lists for training scores, cv scores, fit times (on training data),
    # and the resampling times, which will all be added after each iteration
    cv_results["train_scores"] = []
    cv_results["cv_scores"] = []
    cv_results["cv_fit_times"] = []
    # if no resampler, then set resampling times and resamplined shapes to None
    if resampler is None:
        cv_results["resampling_times"] = None
        cv_results["resampled_shapes"] = None
    else:
        cv_results["resampling_times"] = []
        cv_results["resampled_shapes"] = []
    # training and cv shapes
    cv_results["train_shapes"], cv_results["cv_shapes"] = [], []
    # use KFold to get splitting indices from the returned generator
    kfs = KFold(n_splits = cv, shuffle = shuffle, random_state = random_state)
    splits = kfs.split(X_train, y_train)
    # start, ending times for timing resampling and training
    time_a, time_b = None, None
    # potential best estimator and its score
    best_est, best_score = None, None
    # for each of the set of cv split indices
    for train_index, val_index in splits:
        # training data (not copied)
        X_rs = X_train.loc[train_index, :]
        y_rs = y_train.loc[train_index, :]
        # validation set (no copy)
        X_val = X_train.loc[val_index, :]
        y_val = y_train.loc[val_index, :]
        # add shapes of training and cv data
        cv_results["train_shapes"].append(X_rs.shape)
        cv_results["cv_shapes"].append(X_rs.shape)
        ## resample X_rs, y_rs depending on the resampler type
        if resampler is None: pass
        # already checked if fit_resample as implemented
        elif isinstance(resampler, ABCMeta):
            # instantiate with kwargs and make it point to object instead
            if resampler_kwargs is None: resampler = resampler()
            else: resampler = resampler(**resampler_kwargs)
            # resample, time in seconds
            time_a = time()
            X_rs, y_rs = resampler.fit_resample(X_rs, y_rs)
            time_b = time()
        # else is one of the supported resampling methods (resampler is str)
        elif resampler in _samplers:
            # instantiate with/without keyword args as appropriate
            if resampler_kwargs is None:
                resampler = _samplers[resampler]()
            else: resampler = _sampler[resampler](**resampler_kwargs)
            # resample, time in seconds
            time_a = time()
            X_rs, y_rs = resampler.fit_resample(X_rs, y_rs)
            time_b = time()
        # else is a user-defined function
        elif hasattr(resampler, "__call__"):
            # call with kwargs if resampler_kwargs is not None
            time_a, time_b = None, None
            if resampler_kwargs is None:
                time_a = time()
                X_rs, y_rs = resampler(X_rs, y_rs)
                time_b = time()
            else:
                time_a = time()
                X_rs, y_rs = resampler(X_rs, y_rs, **resampler_kwargs)
                time_b = time()
        # else RuntimeError
        else:
            RuntimeError("{0}: error: unknown resampler type {1}"
                         "".format(_fn, resampler))
        # convert X_rs into DataFrame and write resampling time and shape of
        # resampled data into cv_results
        # if the resampler is not None (X_rs may not be DataFrame)
        if resampler is not None:
            X_rs = DataFrame(X_rs, columns = X_train.columns)
            cv_results["resampling_times"].append(time_b - time_a)
            cv_results["resampled_shapes"].append(X_rs.shape)
        ## fit model on [resampled] training data X_rs, validate with X_val
        # need to clone model so that we are maintaining separate models
        fitted_est = clone(est)
        time_a = time()
        fitted_est = fitted_est.fit(X_rs, y_rs)
        time_b = time()
        # add training time, training score, and validation score
        fitted_cv_score = fitted_est.score(X_val, y_val)
        cv_results["train_times"].append(time_b - time_a)
        cv_results["train_scores"].append(fitted_est.score(X_rs, y_rs))
        cv_results["cv_scores"].append(fitted_cv_score)
        # choose the best estimator based on cv scores; if there are no cv
        # scores (best_est is None), then make fitted_est the best + save score
        if (best_est is None) or \
           ((best_est is not None) and (fitted_cv_score < best_score)):
            best_est, best_score = fitted_est, fitted_cv_score
        # proceed to next cv iteration
    # after resampling/fitting all estimators, getting training/cv scores,
    # training + resampling times, data shapes, and the best estimator, conclude
    # first write in the best estimator (best_est) and cv score (best_score)
    cv_results["best_estimator"] = best_est
    cv_results["best_cv_score"] = best_score
    # compute mean and standard deviation of scores (note ddof = 1)
    cv_results["mean_cv_score"] = mean(cv_results["cv_scores"])
    cv_results["std_cv_score"] = std(cv_results["cv_scores"], ddof = 1)
    # record function ending time
    time_end = time()
    # add overall running time to dict and return
    cv_results["total_time"] = time_end - time_start
    return cv_results

# main
if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
