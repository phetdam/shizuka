# source file containing base classes and methods for used in shizuka
#
# Changelog:
#
# 06-24-2020
#
# removed _MODULE_NAME; just use __module__ instead. renamed shizukaAbstractCV
# to shizukaCoreCV because technically it's not abstract anymore. rewrote
# docstrings for shizukaCoreCV, shizukaBaseCV, and retooled their __repr__
# functions, created several readonly properties for them, and made them inherit
# from touketsu.FrozenClass. added _raw__repr__ method to shizukaCoreCV to
# reduce copy-pasting necessary for creating good representations. made new
# check in shizukaCoreCV to ensure that resampler methods are instance methods,
# using utils.is_method.
#
# todo: rewrite docstring and clean up shizukaSearchCV class and methods.
#
# 06-23-2020
#
# changed base class from ABCMeta to concrete base class. working on docstring.
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
__doc__ = """Base code for the ``shizuka`` package."""

from inspect import getfullargspec
from numpy import mean, std
from pandas import DataFrame
from sys import stderr
from textwrap import fill

from touketsu import FrozenClass # note: may not be uploaded to PyPI yet
from .utils import is_method

class shizukaCoreCV(FrozenClass):
    """Base class that all ``shizuka`` CV results classes inherit from.

    Do not use directly.

    .. note::

       :meth:`self._freeze` needs to be manually called in the :meth:`__init__`
       method of any :class:`shizukaCoreCV` subclass.

    Defines several common properties shared by all the subclasses. Note that
    the :attr:`cv_results` property is a dict and that subclasses may have
    :attr:`cv_results` with differing keys.

    :param best_estimator: Best fitted compatible estimator by validation score.
        Please see :doc:`../doc/source/model_compat` for details on how the term
        "compatible" is being used in this context.
    :type best_estimator: object
    :param cv_results: Per-fold validation results from cross-validation.
    :type cv_results: dict
    :param cv_iter: Number of validation folds/iterations.
    :type cv_iter: int
    :param total_time: Total runtime of cross-validation routine in seconds
    :type total_time: int
    :param shuffle: Indicates if data was shuffled before being split into
        training and validation folds.
    :type shuffle: bool
    :param random_state: ``None`` or integer seed, if any, used for k-fold data
        splitting.
    :type random_state: NoneType, int
    :param resampler: Class instance implementing :meth:`fit_resample` and
        :meth:`get_params` in the vein of ``imblearn``, or a custom resampling
        function. Default ``None``.

        .. note::

           Elaborate more on what "custom resampling function" means and
           give an explicit call signature for :meth:`fit_resample` and 
           :meth:`get_params`.

    :type resampler: object or function, optional
    :param resampler_kwargs: Keyword arguments passed to ``resampler``. Set to
        ``None`` if ``resampler`` is ``None``. Default ``None``.
    :type resampler_kwargs: dict, optional
    """
    def __init__(self, best_estimator, cv_results, cv_iter, total_time, shuffle,
                 random_state, resampler = None, resampler_kwargs = None):
        self._best_estimator = best_estimator
        self._cv_results = cv_results
        # get parameters from best_estimator
        self._best_params = self.best_estimator.get_params()
        self._cv_iter = cv_iter
        self._total_time = total_time
        self._shuffle = shuffle
        self._random_state = random_state
        self._resampler = resampler
        # if resampler is None, ignore value of resampler_kwargs (set to None)
        if self._resampler is None: self._resampler_params = None
        # else if resampler implements fit_resample and get_params, call the
        # get_params method to retrieve the resampler's parameters. we also
        # explicitly check that the methods are instance methods using
        # utils.is_method, which uses type() to check the state of the function.
        elif hasattr(self._resampler, "fit_resample") and \
             is_method(self._resampler.fit_resample) and \
             hasattr(self._resampler, "get_params") and \
             is_method(self._resampler.get_params):
            self._resampler_params = self._resampler.get_params()
        # else if resampler is a resampling function (should have two unnamed
        # with no defaults; optional keyword args allowed. check skipped!)
        elif hasattr(self._resampler, "__call__"):
            self._resampler_params = resampler_kwargs
        else:
            raise TypeError("{0}: resampler must be class instance implementing"
                            " fit_resample and get_params, None, or a function"
                            "".format(self.__init__.__name__))

    def _raw__repr__(self):
        """Raw textual representation of the :class:`shizukaCoreCV`.

        Gives a string representation of a :class:`shizukaCoreCV` instance.
        Output is in the scikit-learn style, i.e. of the form

        .. code:: python

           shizukaCoreCV(best_estimator=LogisticRegression, ...

        This is the raw, unwrapped representation for the class instance.

        .. note::

           Although the representation contains most of the arguments passed to
           :meth:`__init__`, it omits ``cv_results``. This is because
           ``cv_results`` could possibly be very large in some subclasses and
           therefore be unwieldy to display to the screen. Omitting
           ``cv_results`` also gives room to append additional subclass
           attributes to the end of the string generated by :meth:`__repr__`.

        :returns: Raw, unwrapped string representation of the
            :class:`shizukaCoreCV` instance in a scikit-learn like format
        :rtype: str
        """
        # get name of the best estimator using name attribute of class type
        best_est_name = self.best_estimator.__class__.__name__
        resampler_name = "None" if self.resampler is None else \
            self.resampler.__class__.__name__
        # start building the output string and add best_est + params
        out_str = self.__class__.__name__ + "(best_estimator=" + best_est_name \
            + ", " + "best_params=" + repr(self._best_params) + ", "
        # add cv_iter, shuffle, random_state, resampler and the resampler's
        # parameters (can be None), total time, and cv_results.
        return out_str + "cv_iter=" + str(self._cv_iter) + ", " + "shuffle=" + \
            str(self._shuffle) + ", random_state=" + str(self._random_state) + \
            ", resampler=" + resampler_name + ", " + "resampler_params=" + \
            repr(self._resampler_params) + ", " "total_time=" + \
            str(self._total_time) + ")"
        
    def __repr__(self):
        """Textual representation of the :class:`shizukaCoreCV`.
        
        Gives a wrapped string representation of a :class:`shizukaCoreCV`
        instance, essentially just wrapping the output of :meth:`_raw__repr__`
        with :func:`textwrap.fill`.

        :returns: Wrapped string representation of the :class:`shizukaCoreCV`
            instance in a scikit-learn like format
        :rtype: str
        """
        # use _raw__repr__ to get unwrapped representation.
        out_str = self._raw__repr__()
        # use len(self.__class__.__name__) + 1 to determine value of the
        # subsequent_indent parameter. use textwrap.fill to wrap the text
        # and join lines together at the end
        return fill(out_str, width = 80, subsequent_indent = " " * \
                    (len(self.__class__.__name__) + 1))

    def __str__(self): return self.__repr__()

    @property
    def best_estimator(self):
        """Best fitted scikit-learn compatible estimator, by validation score.

        .. note::

           The term "compatible" means that the API for fitting and scoring the
           estimator is consistent with that of the scikit-learn API. See the
           description for ``best_estimator`` in the class docstring for more
           details.

        :rtype: object
        """
        return self._best_estimator

    @property
    def best_params(self):
        """Hyperparameters of :attr:`best_estimator`.

        :rtype: dict
        """
        return self._best_params

    @property
    def cv_iter(self):
        """Number of validation folds/iterations used in cross-validation.

        :rtype: int
        """
        return self._cv_iter

    @property
    def shuffle(self):
        """Indicator if data was shuffled before splitting.
        
        :returns: Whether or not data was shuffled before splitting. ``True``
            indicates that data was shuffled before cross-validation splitting
            while ``False`` indicates no data shuffling.
        :rtype: bool
        """
        return self._shuffle

    @property
    def random_state(self):
        """Seed, if any, used when splitting the data before cross-validation.

        :rtype: int or None
        """
        return self._random_state

    @property
    def resampler(self):
        """Class instance or function used for rebalancing class proportions.

        See ``resampler`` in the class docstring for more details.

        :rtype: object or function
        """
        return self._resampler

    @property
    def resampler_params(self):
        """Keyword arguments passed to ``resampler``.

        ``None`` if :attr:`resampler` is ``None``.

        :rtype: object, NoneType
        """
        return self._resampler_params

    @property
    def total_time(self):
        """Total running time of cross-validation routine, in seconds.

        :rtype: int
        """
        return self._total_time

    @property
    def cv_results(self):
        """Per-fold validation results.

        For ``k`` folds, each key corresponds to an iterable of length ``k``,
        where each value at index ``i`` of the iterable, using zero-indexing,
        corresponds to a result for the ``(i + 1)``th cross-validation
        iteration.

        :returns: Per-fold cross validation results.
        :rtype: dict
        """
        return self._cv_results
    

class shizukaBaseCV(shizukaCoreCV):
    """Class holding model results from k-fold cross-validation routines.

    Contains results for a model trained with k-fold cross-validation and
    optional resampling.

    .. note::

       :meth:`__init__` does not have much type checking, so manual type
       checking is required before creating an instance of this class. Users
       should never need to create a :class:`shizukaBaseCV` instance themselves.

    :param best_estimator: Best fitted compatible estimator in terms of
        validation score. The definition of "compatible" in this context can be
        found in :doc:`../doc/source/model_compat`.
    :type best_estimator: object
    :param cv_results: A dict with per-fold validation results. The keys
        expected to be in the dict and their respective descriptions are below.

        * ``train_scores``: Estimator training scores per fold
        * ``cv_scores``: Estimator validation scores per fold
        * ``train_times``: Training times per fold in seconds
        * ``resampled_shapes``: Shape of resampled training data per fold. If
          ``resampler`` is ``None``, then ``None``.
        * ``cv_shapes``: Shape of validation set, per fold.
    :type cv_results: dict
    :param cv_iter: Number of validation/folds iterations
    :type cv_iter: int
    :param total_time: Total runtime of cross-validation routine in seconds
    :type total_time: int
    :param shuffle: ``True`` if data was shuffled before being split into
        training and validation folds, ``False`` otherwise.
    :type shuffle: bool
    :param random_state: Seed used for k-fold data splitting, if any. ``None``
        if no fixed seed was used.
    :type random_state: int or NoneType
    :param resampler: Class instance implementing :meth:`fit_resample` and
        :meth:`get_params` like the resamplers defined in ``imblearn``, or a
        custom resampling function. Default ``None``.

        .. note::

           See the docstring for :class:`shizukaCoreCV` more for details.

    :type resampler: object or function, optional
    :param resampler_kwargs: Keyword arguments passed to ``resampler``. Ignored
        and set to ``None`` if ``resampler`` is ``None``. Default ``None``.
    :type resampler_kwargs: dict, optional
    """
    def __init__(self, best_estimator, cv_results, cv_iter, total_time, shuffle,
                 random_state, resampler = None, resampler_kwargs = None):
        # call super()
        super().__init__(best_estimator, cv_results, cv_iter, total_time,
                         shuffle, random_state, resampler = resampler,
                         resampler_kwargs = resampler_kwargs)
        # get best_cv_score, mean_cv_score, and std_cv_score from cv_results
        self._best_cv_score = max(self.cv_results["cv_scores"])
        self._mean_cv_score = mean(self.cv_results["cv_scores"])
        # note that standard deviation is calculated with n - 1 denominator here
        self._std_cv_score = std(self.cv_results["cv_scores"], ddof = 1)
        # freeze the class instance
        self._freeze()

    def __repr__(self):
        """Textual representation of the :class:`shizukaBaseCV`.
        
        Gives a wrapped string representation of a :class:`shizukaBaseCV`
        instance. Output is in the scikit-learn style, i.e. with the form

        .. code:: python

           shizukaBaseCV(best_estimator=Perceptron, best_params={'penalty': ...

        :returns: Wrapped string representation of the :class:`shizukaBaseCV`
            instance in a scikit-learn like format
        :rtype: str
        """
        # use _raw__repr__ to get unwrapped representation
        out_str = self._raw__repr__()
        # add metrics and cv_results to the end of out_str
        out_str = out_str[:-1] + ", best_cv_score=" + \
            str(self._best_cv_score) + ", mean_cv_score=" + \
            str(self._mean_cv_score) + ", std_cv_score=" + \
            str(self._std_cv_score) + ", cv_results=" + \
            repr(self._cv_results) + ")"
        # use len(self.__class__.__name__) + 1 to determine value of the
        # subsequent_indent parameter. use textwrap.fill to wrap the text
        # and join lines together at the end
        return fill(out_str, width = 80, subsequent_indent = " " * \
                    (len(self.__class__.__name__) + 1))

    def __str__(self): return self.__repr__()
        
    @property
    def best_cv_score(self):
        """The score of :attr:`best_estimator`, i.e. max of validation scores

        :rtype: float
        """
        return self._best_cv_score

    @property
    def mean_cv_score(self):
        """Average validation score among all the folds.

        :rtype: float
        """
        return self._mean_cv_score

    @property
    def std_cv_score(self):
        """Standard deviation of the validation scores.

        Computed with 1 delta degree of freedom, i.e. :func:`numpy.std` is
        called with ``ddof = 1``.

        :rtype: float
        """
        return self._std_cv_score

    ## decorators for accessing values in self.cv_results as attributes ##
    @property
    def train_scores(self):
        """Per-fold estimator training scores.

        :returns: A list of estimator training scores with length equal to the
            number of cross-validation folds.
        :rtype: list
        """
        return self._cv_results["train_scores"]
    
    @property
    def cv_scores(self):
        """Per-fold estimator validation scores.

        :returns: A list of estimator validation scores with length equal to the
            number of cross-validation folds.
        :rtype: list
        """
        return self._cv_results["cv_scores"]

    @property
    def train_times(self):
        """Per-fold estimator training times.

        :returns: A list of estimator training times in seconds with length
            equal to the number of cross-validation folds.
        :rtype: list
        """
        return self._cv_results["train_times"]

    @property
    def resampling_times(self):
        """Per-fold resampling times.

        :returns: A list of resampling times in seconds with length equal to the
            number of cross-validation folds.
        :rtype: list
        """
        return self._cv_results["resampling_times"]

    @property
    def train_shapes(self):
        """Per-fold training data shapes.

        :returns: A list of training data shapes, each of the form
            ``(n_samples_i, n_features)``, with length equal to the number of
            cross-validation folds.
        :rtype: list
        """
        return self._cv_results["train_shapes"]

    @property
    def resampled_shapes(self):
        """Per-fold training data shapes after resampling.

        :returns: A list of training data shapes after resampling for each fold,
            each of the form ``(n_samples_i, n_features)``, with length equal to
            the number of cross-validation folds.
        :rtype: list
        """
        return self._cv_results["resampled_shapes"]

    @property
    def cv_shapes(self):
        """Per-fold validation data shapes.

        :returns: A list of validation data shapes, each of the form
            ``(n_samples_i', n_features)``, with length equal to the number of
            cross-validation folds.
        :rtype: list
        """
        return self._cv_results["cv_shapes"]

    @property
    def cv_results_df(self):
        """Returns :attr:`cv_results` as a :class:`pandas.DataFrame`.

        Syntactic sugar for returning :attr:`cv_results` attribute as a 
        :class:`pandas.DataFrame`. Each row corresponds to a validation
        iteration, so for ``k`` folds in ``k``-fold cross validation, one would
        get ``k`` rows with the headers

        ::

            |--------------|-----------|-   -|------------------|-----------|
            | train_scores | cv_scores | ... | resampled_shapes | cv_shapes |
            |--------------|-----------|-   -|------------------|-----------|
        """
        return DataFrame(self._cv_results)    

class shizukaSearchCV(shizukaCoreCV):
    """Class with model results from hyperparameter search routines.

    Returned from hyperparameter search routines, containing results for a model
    trained with k-fold cross-validation and optional resampling. The best
    model is defined as the one that has the highest average validation score.

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
    """
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

        # call super() with appropriate parameters
        super().__init__(best_estimator, cv_results, cv_iter, total_time,
                         shuffle, random_state, resampler = resampler,
                         resampler_kwargs = resampler_kwargs)
        # get best_cv_score (best mean cv score). note best_resampler and
        # best_resampler_params are simply decorators for returning the
        # resampler and resampler_params attributes of the abstract parent.
        self.best_cv_score = max(self.cv_results["mean_cv_score"])
        # freeze
        self._freeze()

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
        """Returns :attr:`cv_results` as a :class:`pandas.DataFrame`.

        Syntactic sugar for returning :attr:`cv_results` attribute as a 
        :class:`pandas.DataFrame` with ``n_1 * ... n_m`` models, with a row for
        each model/parameter combination, where there are ``m`` parameters with
        ``n_i`` values for the ``i``th parameter. Note that the number of
        columns may change, depending on the number of cross-validation folds
        chosen and whether resampling was used.

        For example, following the format of the scikit-learn
        :class:`sklearn.model_seklection.GridSearchCV` example, the
        :class:`pandas.DataFrame` could have a header format given by

        ::

          |--------------|-------------|-   -|---------------|--------------|-
          | param_kernel | param_gamma | ... | mean_cv_score | std_cv_score | ...
          |--------------|-------------|-   -|---------------|--------------|-

        """
        return DataFrame(self._cv_results)

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
        # get unwrapped, raw output from _raw__repr__
        out_str = self._raw__repr__()
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

    def __str__(self): return self.__repr__()

if __name__ == "__main__":
    print("{0}: do not run module as script".format(__module__), file = stderr)
