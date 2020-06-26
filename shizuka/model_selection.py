# implements some useful methods for model selection. intended as a complement
# to the more general methods available in sklearn or other libraries.
#
# Changelog:
# 
# 06-26-2020
#
# add warning about needing dask.array.Array for out-of-core computation + fix
# :doc: reference to module_compat.rst. corrected imports from base.py.
#
# 06-24-2020
#
# removed _MODULE_NAME, using __module__ instead. renamed resampled_cv to
# resampled_fit_cv. started rewriting resampled_fit_cv docstring and added more
# keyword arguments.
#
# 01-25-2020
#
# removed some old commented code that is no longer in use and cleaned up the
# docstring, which still had old comments about the dict that used be returned.
#
# 01-23-2020
#
# corrected premature conversion of single-column DataFrame to ndarray that
# causes label-based indexing to fail in the main loop. fixed accidental removal
# of the best_score intermediate value to help choose best estimator.
#
# 01-21-2020
#
# changed NotImplementedError to AttributeError in resampled_cv. added @property
# decorators to the shizukaBaseCV class and moved it to a new file, base.py.
# making changes to docstring of resampled_cv; retwrote to return shizukaBaseCV.
# added additional type checking for resampler arg, which only needs to
# implement fit_resample and get_params, which is needed for shizukaBaseCV to be
# able to get parameters from the resampling object.
#
# 01-20-2020
#
# added of conversion of y_train vector in resampled_cv to ndarray so that there
# won't be a DataConversionWarning. rewrote resampler initialization to prevent
# repeated instantiation of a resampling object, as that it wasteful. corrected
# error where cv_shapes were given train_shapes values instead. added a first
# draft of shizukaBaseCV, which will most likely be revised later.
#
# 01-19-2020
#
# corrected some miscellaneous syntax and name errors. sucks not to have pylint.
# corrected runtime bug in which resampling would only be applied to the first
# iteration since we overwrite the resampling object.
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
__doc__ = """Contains methods for more specialized model selection.

Not intended as a replacement for a more complete machine learning package like
scikit-learn, but as a complement for more general model selection routines.
"""

from abc import ABCMeta
from numpy import array, ravel
from pandas import DataFrame
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE
from inspect import getfullargspec
from numpy import mean, std
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
from sys import stderr
from time import time

from .base import CoreCVResults, SearchCVResults

## internal constants
# supported internal scoring methods for resampled_cv
_scorings = ["r2", "accuracy"]
# supported internally specified resampling methods for resampled_cv
_samplers = {"SMOTE": SMOTE, "ADASYN": ADASYN, "SMOTEENN": SMOTEENN}

def resampled_fit_cv(est, X_train, y_train, fitter = None, fitter_kwargs = None,
                     predictor = None, predictor_kwargs = None, scorer = None,
                     scorer_kwargs = None, resampler = None,
                     resampler_kwargs = None, shuffle = False, cv = 3,
                     n_jobs = 1, random_state = None):
    """Cross-validated model training, with optional resampling.

    Trains scikit-learn compatible supervised classifiers with cross-validation
    and optional resampling. Avoids the naive error of validating with a fold of
    resampled training data, as here the original training data used in training
    folds is copied and validated on the untouched validation fold. Best used
    for training a single model with or without resampling using k-fold cross-
    validation.

    Using built-in resampling routines requires the package ``imblearn``, which
    can be found on PyPI.

    .. note::

       Standard deviation of cross-validation scores is computed with
       denominator ``k - 1``, if ``k`` is the number of cross-validation folds.

    .. warning::

       ``imblearn`` resamplers only work with data objects castable to
       :class:`numpy.ndarray` and therefore cannot support distributed arrays
       or objects larger than memory. Use ``"dask"`` backend with
       :func:`resampled_fit_cv` and pass in :class:`dask.array.Array` to 
       ``X_train``, ``y_train`` to support out-of-core computation.
       

    :param est: An unfitted compatible or scikit-learn compatible classifier.

        .. note::

           The notion of a model being "compatible" or "scikit-learn compatible"
           is defined in :doc:`..//model_compat`. Please read to
           ensure that whatever is being passed into ``est`` fits either of
           these notions.
    :type est: object
    :param X_train: Training data matrix with shape ``(n_samples, n_features)``,
        castable to :class:`numpy.ndarray`. Must fit in memory. If not of type
        :class:`numpy.ndarray` with float elements, a conversion will be
        performed internally.
    :type X_train: array-like
    :param y_train: Training response matrix/vector with shape ``(n_samples,)``
        or ``(n_samples, n_outputs)``, castable to :class:`numpy.ndarray`. Must
        fit in memory. If not of type :class:`numpy.ndarray`, with float
        elements, a conversion will be performed internally.
    :type y_train: array-like
    :param fitter: The method used to fit ``est`` if ``est`` does not implement
        a ``fit`` method. Even if ``est`` implements a ``fit`` method, if
        ``fitter`` is not ``None``, then ``fitter`` will be called. Must have
        the call signature

        .. code:: python

           (X_train: numpy.ndarray, y_train: numpy.ndarray, **fitter_kwargs) -> object


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

                      note: currently ignored. uses only default scoring.

    resampler         optional resampling class name or function for resampling,
                      or a recognized resampling method such as "SMOTE" or 
                      "SMOTEENN". acceptable resampling classes must implement
                      methods fit_resample and get_params with signatures

                      def fit_resample(self, X, y):
                          \"\"\"
                          parameters:

                          X    shape (n_samples, n_features)
                          y    shape (n_samples,),

                          returns:

                          Xr, yr   shape (n_samples_new, n_features), (n_samples)
                          \"\"\"
                          # code

                      def get_params(self):
                          \"\"\"
                          returns:

                          params    dict "parameter": value
                          \"\"\"
                          # code

                      a user-defined resampling function must have signature

                      def [some_name](X, y, *args, **kwargs):
                          \"\"\"
                          parameters:

                          X    shape (n_samples, n_features)
                          y    shape (n_samples,)
                          *args, **kwargs

                          returns:

                          Xr, yr   shape (n_samples_new, n_features), (n_samples)
                          \"\"\"
                          # code

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
    _fn = resampled_cv.__name__
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
    # implements fit_resample and get_params, with appropriate signatures
    if (resampler is None): pass
    # check fit_resample and get_params signatures
    elif hasattr(resampler, "fit_resample") and \
         hasattr(resampler, "get_params"):
        # gets full arg spec
        fr_spec = getfullargspec(resampler.fit_resample)
        gp_spec = getfullargspec(resampler.get_params)
        # check signature of fit_resample; must have 2 unnamed args only. since
        # this is a class method, we need to ignore the self arg. no varargs.
        # note that .defaults may be None, which we need to check; check "self"
        if ("self" in fr_spec.args) and (fr_spec.varargs == None) and \
           (((fr_spec.defaults is None) and (len(fr_spec.args) - 1 == 2)) or
            (len(fr_spec.args) - len(fr_spec.defaults) - 1 == 2)): pass
        else:
            raise ValueError("{0}: fit_resample must have signature with only "
                             "self, two positional args, >= 0 named args, and "
                             "optional **kwargs".format(_fn))
        if ("self" in gp_spec.args) and (gp_spec.varargs == None) and \
           (((gp_spec.defaults is None) and (len(gp_spec.args) == 1)) or
            (len(gp_spec.args) - len(gp_spec.defaults) == 1)): pass
        else:
            raise ValueError("{0}: get_params must have signature with only "
                             "self, >= 0 named args, and optional **kwargs"
                             "".format(_fn))
    # else if a function (more accurately, is callable), check call signature
    elif hasattr(resampler, "__call__"):
        fas = getfullargspec(resampler)
        # check that there are only two positional args; i.e. length of args -
        # number of defaults is 2 (args includes named args), and does not
        # allow any more positional args. if not, error
        # note: we need to do extra check since fas.defaults may be None, and
        # "self" cannot be in fas.args (else it is a class type)
        if ("self" not in fas.args) and (fas.varargs == None) and \
           (((fas.defaults is None) and (len(fas.args) == 2)) or
            (len(fas.args) - len(fas.defaults) == 2)): pass
        else:
            raise ValueError("{0}: resampling function f must have signature "
                             "with only 2 positional args, >= 0 named args, "
                             "and optional **kwargs".format(_fn))
    # else if it is one of the internally supported sampling methods
    elif isinstance(resampler, str) and (resampler in _samplers): pass
    else:
        raise TypeError("{0}: resampler type must be class implementing "
                        "fit_transform and get_params, a function, or supported "
                        "resampling method in {1}. see docstring for call "
                        "signatures".format(_fn, tuple(_samplers.keys())))
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
    # create results dict for shizukaBaseCV
    cv_results = {}
    # set lists for training scores, cv scores, fit times (on training data),
    # and the resampling times, which will all be added after each iteration
    cv_results["train_scores"] = []
    cv_results["cv_scores"] = []
    cv_results["train_times"] = []
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
    # best estimator and score
    best_est, best_score = None, None
    # if resampler has fit_resample and get_params methods, instantiate
    if hasattr(resampler, "fit_resample") and \
       hasattr(resampler, "get_params"):
        # instantiate with kwargs and make it point to object instead
        if resampler_kwargs is None: resampler = resampler()
        else: resampler = resampler(**resampler_kwargs)
    # else is one of the supported resampling methods (resampler is str)
    elif resampler in _samplers:
        # instantiate with/without keyword args as appropriate
        if resampler_kwargs is None: resampler = _samplers[resampler]()
        else: resampler = _sampler[resampler](**resampler_kwargs)
    ## main loop: for each of the set of cv split indices do
    for train_index, val_index in splits:
        # training data (not copied); flatten and convert to ndarray to suppress
        # sklearn's DataConversionWarning. i.e. expected shape (n,), got (n, 1)
        X_rs = X_train.loc[train_index, :]
        y_rs = array(ravel(y_train.loc[train_index, :]))
        # validation set (no copy)
        X_val = X_train.loc[val_index, :]
        y_val = array(ravel(y_train.loc[val_index, :]))
        # add shapes of training and cv data
        cv_results["train_shapes"].append(X_rs.shape)
        cv_results["cv_shapes"].append(X_val.shape)
        ## resample X_rs, y_rs depending on the resampler type
        # do nothing if resampler is None
        if resampler is None: pass
        # else if resampler has fit_resample, it is a resampling object
        elif hasattr(resampler, "fit_resample"):
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
    # record function ending time
    time_end = time()
    # instantiate new shizukaBaseCV object with results and return
    return shizukaBaseCV(best_est, cv_results, cv, time_end - time_start,
                         shuffle, random_state, resampler = resampler,
                         resampler_kwargs = resampler_kwargs)

def resampled_grid_search_cv(est, X_train, y_train, param_grid, scoring = None,
                             resamplers = None, rs_param_dicts = None,
                             shuffle = False, random_state = None, cv = 3,
                             n_jobs = 1):
    """
    given a hypothesis set denoted by an unfitted estimator, training feature
    matrix and response vector, grid (dict) of parameters for the estimator,
    with optional list of resampling objects/functions and list of grids (dicts)
    of named parameters for each resampling object/function, perform a grid
    search over all possible parameters and resampler + resampler kwarg
    combinations with cross-validation to choose the best set of parameters for
    the model, the best resampler, and the best resampler parameters.

    returns a shizuka.base.shizukaSearchCV instance. details in shizuka.base.

    like resampled_cv, requires imblearn, which can be downloaded from PyPI.

    notes: standard deviation of CV scores is computed iwth denomiator cv - 1.
           since an exhaustive search is performed, it is highly recommended to
           limit the number of grid search parameters for the estimator and
           resamplers.

    parameters:

    est               sklearn regression or classification estimator, a subclass
                      of ClassifierMixin or RegressorMixin from sklearn.base.
                      should be unfitted and initialized with hyperparameters.
    X_train           2d training feature matrix, required to be DataFrame
    y_train           1d training response vector, same number of observations
                      as X_train, ex. ndarray, Series, or one-column DataFrame.
    param_grid        dictionary where each key corresponds to a named parameter
                      in est, with an associated iterable of values to search.
    scoring           optional metric used to compute score of the estimator, 
                      default None which indicates that the estimator's builtin 
                      score function is to be used. function only supports 
                      "accuracy" for classifiers and "r2" for regressors.

                      note: currently ignored. uses only default scoring.
    """
    pass

# main
if __name__ == "__main__":
    print("{0}: do not run module as script".format(__module__), file = stderr)
