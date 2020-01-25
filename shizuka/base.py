# source file containing base classes and methods for used in shizuka
#
# Changelog:
#
# 01-23-2020
#
# finally wrote the __repr__() method for shizukaBaseCV.
#
# 01-21-2020
#
# initial creation. moved shizukaBaseCV from model_selection.py, and corrected
# confusion between abstract base class and object subclass type. streamlined
# some type checking for the resampler, which should be a class instance.
#
__doc__ = "base code for the shizuka package"

_MODULE_NAME = "shizuka.base"

from abc import ABCMeta
from inspect import getfullargspec
from numpy import mean, std
from pandas import DataFrame
from sys import stderr
from textwrap import fill

class shizukaBaseCV:
    """
    class returned from cross-validation routines containing results for a model
    trained with k-fold cross-validation and optional resampling.

    note: does not contain a lot of type-checking, so manual type-checking is
          required when instantiating an instance of the class. users should
          never need to create an instance of shizukaBaseCV themselves.

    attributes:

    best_estimator    best fitted sklearn estimator based on validation score
    best_params       [hyper]parameters of best_estimator
    best_cv_score     validation score of best_estimator
    mean_cv_score     average of validation scores for each fold
    std_cv_score      sample standard deviation of validation scores; computed
                      with ddof = 1, so denominator k - 1 for k folds
    cv_iter           number of validation folds/iterations
    shuffle           boolean, indicates if data was shuffled before splitting
    random_state      seed (if any) used for data splitting
    resampler         None, class instance implementing fit_resample and 
                      get_params as detailed in the docstring of shizuka
                      .model_selection.resampled_cv, or a function
    resampler_params  None or dict. if resampler is class instance implementing
                      fit_resample and get_params, then the value will be the
                      dict returned by the object's get_params call. if the
                      resampler is a function, then the value will be the dict
                      of any keyword arguments passed to it.
    total_time        total running time of cross-validation routine in seconds
    cv_results        dict with validation results for each fold. keys:

                      train_scores      estimator training scores per fold
                      cv_scores         estimator validation scores per fold
                      train_times       training times per fold in seconds
                      resampling_times  resampling times per fold in seconds
                                        or None if resampler is None
                      train_shapes      shape of training set per fold
                      resampled_shapes  shape of resampled training data per
                                        fold; None if resampler is None
                      cv_shapes         shape of validation set per fold

                      dict values may also be accessed directly as attributes,
                      which is the syntactically preferred method of access.
    """
    def __init__(self, best_estimator, cv_results, cv_iter, total_time,
                 shuffle, random_state, resampler = None,
                 resampler_kwargs = None):
        """
        constructor for shizukaBaseCV

        parameters:

        best_estimator    best fitted sklearn estimator, by validation score
        cv_results        dict with validation results for each fold.

                          train_scores      estimator training scores per fold
                          cv_scores         estimator validation scores per fold
                          train_times       training times per fold in seconds
                          resampling_times  resampling times per fold in seconds
                                            or None if resampler is None
                          train_shapes      shape of training set per fold
                          resampled_shapes  shape of resampled training data per
                                            fold; None if resampler is None
                          cv_shapes         shape of validation set per fold

        cv_iter           number of validation folds/iterations
        total_time        total runtime of cross-validation routine in seconds
        shuffle           boolean, indicates if data was shuffled before being
                          split into training and validation folds.
        random_state      None or seed (if any) used for k-fold data splitting
        resampler         optional, default None. None, class instance
                          implementing fit_resample and get_params, or 
                          a custom resampling function.
        resampler_kwargs  optional, default None. if resampler is not None,
                          resampler_kwargs gives keyword args passed to the
                          resampler (function, abc.ABCMeta), else ignored.
        """
        self.best_estimator = best_estimator
        self.cv_results = cv_results
        # get parameters from best_estimator
        self.best_params = self.best_estimator.get_params()
        # get best_cv_score, mean_cv_score, and std_cv_score from cv_results
        self.best_cv_score = max(self.cv_results["cv_scores"])
        self.mean_cv_score = mean(self.cv_results["cv_scores"])
        # note that standard deviation is calculated with n - 1 denominator here
        self.std_cv_score = std(self.cv_results["cv_scores"], ddof = 1)
        self.cv_iter = cv_iter
        self.total_time = total_time
        self.shuffle = shuffle
        self.random_state = random_state
        self.resampler = resampler
        # if resampler is None, ignore value of resampler_kwargs (set to None)
        if self.resampler is None: self.resampler_params = None
        # else if resampler implements fit_resample and get_params, call the
        # get_params method to retrieve the resampler's parameters. we also
        # explicitly check that the methods are instance methods.
        # note: __init__ is required since we cannot call getfullargspec on
        # a class instance by itself. ir resampler was a class name, then ok
        elif ("self" in getfullargspec(self.resampler.__init__).args) and \
             hasattr(self.resampler, "fit_resample") and \
             hasattr(self.resampler, "get_params"):
            self.resampler_params = self.resampler.get_params()
        # else if resampler is a resampling function (should have two unnamed
        # with no defaults; optional keyword args allowed. check skipped!)
        elif hasattr(self.resampler, "__call__"):
            self.resampler_params = resampler_kwargs
        else:
            raise TypeError("{0}: resampler must be class instance implementing"
                            " fit_resample and get_params, None, or a function"
                            "".format(self.__init__.__name__))

    ## decorators for accessing values in self.cv_results as attributes ##
    @property
    def train_scores(self): return self.cv_results["train_scores"]
    
    @property
    def cv_scores(self): return self.cv_results["cv_scores"]

    @property
    def train_times(self): return self.cv_results["train_times"]

    @property
    def resampling_times(self): return self.cv_results["resampling_times"]

    @property
    def train_shapes(self): return self.cv_results["train_shapes"]

    @property
    def resampled_shapes(self): return self.cv_results["resampled_shapes"]

    @property
    def cv_shapes(self): return self.cv_results["cv_shapes"]

    @property
    def cv_results_df(self):
        """
        syntactic sugar to return cv_results attribute as a pandas.DataFrame. 
        each row corresponds to a validation iteration. so for k folds in
        k-fold cross validation, one would get k rows with headers 

        |--------------|-----------|-   -|------------------|-----------|
        | train_scores | cv_scores | ... | resampled_shapes | cv_shapes |
        |--------------|-----------|-   -|------------------|-----------|
        """
        return DataFrame(self.cv_results)

    ## instance methods ##
    def __repr__(self):
        """
        defines textual representation of shizukaBaseCV. text wrapping is built
        in to prevent input from flying off the screen.
        """
        # get name of the best estimator using name attribute of class type
        best_est_name = self.best_estimator.__class__.__name__
        resampler_name = "None" if self.resampler is None else \
            self.resampler.__class__.__name__
        # start building the output string and add best_est + params
        out_str = self.__class__.__name__ + "(best_estimator=" + best_est_name + ", " + \
            "best_params=" + repr(self.best_params) + ", "
        # add metrics, cv_iter, shuffle, random_state, resampler and the
        # resampler's parameters (can be None), total time, and cv_results
        out_str = out_str + "best_cv_score=" + str(self.best_cv_score) + \
            ", mean_cv_score=" + str(self.mean_cv_score) + ", std_cv_score=" + \
            str(self.std_cv_score) + ", cv_iter=" + str(self.cv_iter) + ", " + \
            "shuffle=" + str(self.shuffle) + ", random_state=" + \
            str(self.random_state) + ", resampler=" + resampler_name + ", " + \
            "resampler_params=" + repr(self.resampler_params) + ", " \
            "total_time=" + str(self.total_time) + ", cv_results=" + \
            repr(self.cv_results) + ")"
        # use len(self.__class__.__name__) + 1 to determine value of the
        # subsequent_indent parameter. use textwrap.fill to wrap the text
        # and join lines together at the end
        return fill(out_str, width = 80, subsequent_indent = " " * \
                    (len(self.__class__.__name__) + 1))

    def __str__(self):
        """returns self.__repr__()"""
        return self.__repr__()

if __name__ == "__main__":
    print("{0}: do not run module as script".format(_MODULE_NAME),
          file = stderr)
