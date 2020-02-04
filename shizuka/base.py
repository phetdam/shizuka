# source file containing base classes and methods for used in shizuka
#
# Changelog:
#
# 02-04-2020
#
# finished a first prototype of shizukaSearchCV.
#
# 02-03-2020
#
# added docstring for shizukaSearchCV and decorators for cv_results entries that
# are not dependent on the number of folds present. corrected the accidental
# inclusion of best_cv_score as an attribute for shizukaAbstractCV.
#
# 02-02-2020
#
# made some minor code formatting changes to shizukaBaseCV and added a skeleton
# __init__ method for the shizukaSearchCV class.
#
# 01-27-2020
#
# added shizukaAbstractCV as the abstract base class that all shizuka*CV
# subclasses inherit from in order to organize the class hierarchy better.
# edited shizukaBaseCV to inherit from shizukaAbstractCV. need to test.
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

from abc import ABCMeta, abstractmethod
from inspect import getfullargspec
from numpy import mean, std
from pandas import DataFrame
from sys import stderr
from textwrap import fill

class shizukaAbstractCV(metaclass = ABCMeta):
    """
    base class that all shizuka*CV classes inherit from. defines several common
    properties shared by all the subclasses. note that the cv_results property
    is a dict, and the subclasses may have cv_results with differing keys.

    (abstract) attributes:

    best_estimator    best fitted sklearn estimator based on validation score
    best_params       [hyper]parameters of best_estimator
    cv_iter           number of validation folds/iterations
    shuffle           boolean, indicates if data was shuffled before splitting
    random_state      seed (if any) used for data splitting
    resampler         None, class instance implementing fit_resample and
                      get_params with signatures as detailed in the docstring
                      of shizuka.model_selection.resampled_cv, or a function
    resampler_params  None or dict. if resampler is class instance implementing
                      fit_resample and get_params, then the value will be the
                      dict returned by the object's get_params call. if the
                      resampler is a function, then the value will be the dict
                      of any keyword arguments passed to it.
    total_time        total running time of cross-validation routine in seconds
    cv_results        dict containing per-fold validation results. from a high
                      level, for k folds, each key is an iterable of length k
                      where each value at position i in the iterable is a result
                      for the (i + 1)th cross-validation iteration.
    """
    @abstractmethod
    def __init__(self, best_estimator, cv_results, cv_iter, total_time,
                 shuffle, random_state, resampler = None,
                 resampler_kwargs = None):
        """
        abstract constructor. all subclasses are required to override this, and
        will most likely call super().__init__(...) in their own constructors.

        does not contain much type checking.

        parameters:

        best_estimator    best fitted sklearn estimator, by validation score
        cv_results        dict with per-fold validation results.
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

    # require subclasses to override __repr__ since all subclasses should have
    # representations that are meaningful and informative to the user
    @abstractmethod
    def __repr__(self): pass
    

class shizukaBaseCV(shizukaAbstractCV):
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
                      get_params as detailed in the docstring of shizuka.
                      model_selection.resampled_cv, or a function
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
    def __init__(self, best_estimator, cv_results, cv_iter, total_time, shuffle,
                 random_state, resampler = None, resampler_kwargs = None):
        """
        constructor for shizukaBaseCV, overriding that of shizukaAbstractCV

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
        # call super()
        super().__init__(best_estimator, cv_results, cv_iter, total_time,
                         shuffle, random_state, resampler = resampler,
                         resampler_kwargs = resampler_kwargs)
        # get best_cv_score, mean_cv_score, and std_cv_score from cv_results
        self.best_cv_score = max(self.cv_results["cv_scores"])
        self.mean_cv_score = mean(self.cv_results["cv_scores"])
        # note that standard deviation is calculated with n - 1 denominator here
        self.std_cv_score = std(self.cv_results["cv_scores"], ddof = 1)

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

        output is in the sklearn repr style, i.e.

        shizukaBaseCV(best_estimator=LogisticRegression, best_params={'C': ...
        """
        # get name of the best estimator using name attribute of class type
        best_est_name = self.best_estimator.__class__.__name__
        resampler_name = "None" if self.resampler is None else \
            self.resampler.__class__.__name__
        # start building the output string and add best_est + params
        out_str = self.__class__.__name__ + "(best_estimator=" + best_est_name \
            + ", " + "best_params=" + repr(self.best_params) + ", "
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


class shizukaSearchCV(shizukaAbstractCV):
    """
    class returned from hyperparameter search routines containing results for a
    model trained with k-fold cross-validation and optional resampling. the best
    model is defined as the one that has the highest average validation score.

    attributes:

    best_estimator         best fitted sklearn estimator by average cv score per
                           validation fold. the final best_estimator is fit on
                           all the training data with the parameters given by
                           best_params that were the best performing on average
                           and the resampler given by best_resampler (if any)
                           with parameters best_resampler (if any).
    best_params            [hyper]parameters of best_estimator
    best_cv_score          mean cv (validation) score of best_estimator
    best_resampler         None, class instance implementing fit_resample and
                           get_params as detailed in the docstring of shizuka.
                           model_selection.resampled_cv, or a function.
                           resampler used (if any) with best_estimator.
    best_resampler_params  None or dict. if resampler is class instance that
                           implements fit_resample and get_params, then value
                           will be the dict returned by object's get_params
                           method. if the resampler is a function, then value
                           will be the dict of any keyword arguments passed.
                           params for resampler used with best_estimator.
    cv_iter                number of validation folds/iterations
    shuffle                boolean, indicates if data was shuffled before splits
    random_state           seed (if any) used for data splitting
    total_time             total running time of the search routine in seconds
    cv_results             dict with validation results for each model. for each
                           of the m models trained, each key points to a list of
                           values where the ith index of each list corresponds
                           to the ith model trained in the search method. keys:

                           param_name         value of param "name" per model.
                                              number of column varies with
                                              parameters passed to search.
                           resampler          string representing name of the
                                              resampling class instance/function
                                              used for each model
                           rs_param_name      value of resampler param "name"
                                              for each model. may not have these
                                              columns if resampler has no kwargs
                                              or have multiple columns.
                           foldk_cv_score     model validation scores for the
                                              kth validation fold, total of 
                                              cv_iter columns fo cv scores
                           mean_cv_score      mean model validation score
                           std_cv_score       sample standard deviation of model
                                              validation scores; computed with
                                              ddof = 1. so for k validation
                                              folds, use denominator k - 1.
                           rank_cv_score      model ranking by average cv score
                           foldk_train_score  model training scores for the kth
                                              validation fold. total of cv_iter
                                              columns for train scores.
                           mean_train_score   mean model training score
                           std_train_score    sample standard deviation of model
                                              training scores with ddof = 1.
                           mean_train_time    mean model training times, seconds
                           mean_rs_time       mean resampling times, seconds
                           foldk_train_shape  shape of kth fold training set,
                                              total of cv_iter columns
                           foldk_rs_shape     shape of kth fold training set
                                              after application of resampling.
                                              if no resampling was introduced,
                                              may not be present, else cv_iter
                                              total columns with resampling.
                           foldk_cv_shape     shape of kth fold validation set,
                                              total of cv_iter columns.
    
                           dict values resampler, mean_cv_score, std_cv_score,
                           rank_cv_score, mean_train_score, std_train_score,
                           mean_train_time, mean_rs_time may also be accessed
                           directly as instance attributes, except each will
                           end in an s, i.e. cv_results["resampler"] can be
                           directly accessed as resamplers. this is preferable
                           to constantly typing cv_results[some_name].
    """
    def __init__(self, best_estimator, cv_results, cv_iter, total_time, shuffle,
                 random_state, resampler = None, resampler_kwargs = None):
        """
        best_estimator     best fitted sklearn estimator by average cv score
                           per validation fold, fit on all the training data
                           with the parameters that were best performing on
                           average and the best resampler given by resampler (if
                           any) with best resampler_kwargs (if any).
        cv_results         dict with validation results for each model. for each
                           of the m models trained, each key points to a list of
                           values where the ith index of each list corresponds
                           to the ith model trained in the search method. keys:

                           param_name         value of param "name" per model.
                                              number of column varies with
                                              parameters passed to search.
                           resampler          string representing name of the
                                              resampling class instance/function
                                              used for each model
                           rs_param_name      value of resampler param "name"
                                              for each model. may not have these
                                              columns if resampler has no kwargs
                                              or have multiple columns.
                           foldk_cv_score     model validation scores for the
                                              kth validation fold, total of 
                                              cv_iter columns fo cv scores
                           mean_cv_score      mean model validation score
                           std_cv_score       sample standard deviation of model
                                              validation scores; computed with
                                              ddof = 1. so for k validation
                                              folds, use denominator k - 1.
                           rank_cv_score      model ranking by average cv score
                           foldk_train_score  model training scores for the kth
                                              validation fold. total of cv_iter
                                              columns for train scores.
                           mean_train_score   mean model training score
                           std_train_score    sample standard deviation of model
                                              training scores with ddof = 1.
                           mean_train_time    mean model training times, seconds
                           mean_rs_time       mean resampling times, seconds
                           foldk_train_shape  shape of kth fold training set,
                                              total of cv_iter columns
                           foldk_rs_shape     shape of kth fold training set
                                              after application of resampling.
                                              if no resampling was introduced,
                                              may not be present, else cv_iter
                                              total columns with resampling.
                           foldk_cv_shape     shape of kth fold validation set,
                                              total of cv_iter columns.
        cv_iter            number of validation folds/iterations
        total_time         total running time of the search routine in seconds
        shuffle            boolean, indicates if data was shuffled before being
                           split into training and validation folds
        random_state       None or seed (if any) used for k-fold data splitting
        resampler          None, class instance implementing fit_resample and
                           get_params as detailed in the docstring of shizuka.
                           model_selection.resampled_cv, or a function.
                           resampler used (if any) with best_estimator.
        resampler_params   None or dict. if resampler is class instance that
                           implements fit_resample and get_params, then value
                           will be the dict returned by object's get_params
                           method. if the resampler is a function, then value
                           will be the dict of any keyword arguments passed.
                           params for resampler used with best_estimator.
        """
        # call super() with appropriate parameters
        super().__init__(best_estimator, cv_results, cv_iter, total_time,
                         shuffle, random_state, resampler = resampler,
                         resampler_kwargs = resampler_kwargs)
        # get best_cv_score (best mean cv score). note best_resampler and
        # best_resampler_params are simply decorators for returning the
        # resampler and resampler_params attributes of the abstract parent.
        self.best_cv_score = max(self.cv_results["mean_cv_score"])

    ## attribute decorators ##
    @property
    def best_resampler(self): return self.resampler

    @property
    def best_resampler_params(self): return self.resampler_params

    @property
    def resamplers(self): return self.cv_results["resampler"]

    @property
    def mean_cv_scores(self): return self.cv_results["mean_cv_score"]

    @property
    def std_cv_scores(self): return self.cv_results["std_cv_score"]

    @property
    def rank_cv_scores(self): return self.cv_results["rank_cv_score"]

    @property
    def mean_train_scores(self): return self.cv_results["mean_train_score"]

    @property
    def std_train_scores(self): return self.cv_results["std_train_score"]

    @property
    def mean_train_times(self): return self.cv_results["mean_train_time"]
        
    @property
    def mean_rs_times(self): return self.cv_results["mean_rs_time"]
        
    @property
    def cv_results_df(self):
        """
        syntactic sugar to return cv_results attribute as a pandas.DataFrame
        with n_1 * ... n_m models, with a row for each model/parameter
        combination, where there are m parameters with n_i values for the ith
        parameter. note that the number of columns may change, depending on the
        number of cross-validation folds chosen and whether resampling was used.

        for example, following the format of the sklearn GridSearchCV example,
        the DataFrame could have a header format given by

        |--------------|-------------|-   -|---------------|--------------|-
        | param_kernel | param_gamma | ... | mean_cv_score | std_cv_score | ...
        |--------------|-------------|-   -|---------------|--------------|-
        """
        return DataFrame(self.cv_results)

    def __repr__(self):
        """
        defines textual representation of shizukaSearchCV. text wrapping built
        in to prevent input from flying off the screen. unlike shizukaBaseCV,
        cv_results is not shown, as it may be too large to be meaningfully read.
        instead, only the columns resampler, mean_cv_score, std_cv_score,
        rank_cv_score, mean_train_score, std_train_score, mean_train_time, and 
        mean_rs_time will be shown, and as if they were instance attributes.
        that is, for example, cv_results["rank_cv_score"] will appear as an
        attribute rank_cv_scores, in the style of decorators for cv_results.

        output is in the sklearn repr style, i.e.

        shizukaSearchCV(... tbd
        """
        # get name of the best estimator using name attribute of class type
        best_est_name = self.best_estimator.__class__.__name__
        best_resampler_name = "None" if self.best_resampler is None else \
            self.best_resampler.__class__.__name__
        # start building the output string and add best_estimator, best_params,
        # best_cv_score, best_resampler, best_resampler_params, cv_iter,
        # shuffle, random_state, total_time to out_str; note that cv_results is
        # dict but is not fully displayed
        out_str = self.__class__.__name__ + "(best_estimator=" + best_est_name \
            + ", " + "best_params=" + repr(self.best_params) + ", " + \
            "best_cv_score=" + str(self.best_cv_score) + ", best_resampler=" \
            + str(self.best_resampler) + ", best_resampler_params=" + \
            repr(self.best_resampler_params) + ", cv_iter=" + \
            str(self.cv_iter) + ", shuffle=" + str(self.shuffle) + ", " + \
            "random_state=" + str(self.random_state) + ", total_time=" + \
            str(self.total_time) + ", cv_results=dict, "
        # add the columns in cv_results that can be accessed as attributes
        out_str = out_str + "resamplers=" + repr(self.resamplers) + ", " + \
            "mean_cv_scores=" + repr(self.mean_cv_scores) + ", std_cv_scores=" \
            + repr(self.std_cv_scores) + ", rank_cv_scores=" + \
            repr(self.rank_cv_scores) + ", mean_train_scores=" + \
            repr(self.mean_train_scores) + ", std_train_scores=" + \
            repr(self.std_train_scores) + ", mean_train_times=" + \
            repr(self.mean_train_times) + ", mean_rs_times=" + \
            repr(self.mean_rs_times) + ")"
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
