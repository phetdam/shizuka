# source file containing base classes and methods for used in shizuka
#
# Changelog:
#
# 06-26-2020
#
# change n_res_samples_i to n_rs_samples_i to keep consistent format. renamed
# all classes to omit the "shizuka" prefix and gain the "Results" suffix. added
# best_score and scorer_name parameters to all the class constructors as well
# as properties for the two in BaseCVResults. replace list types with
# numpy.ndarray for more efficiency + ability to take advantage of vectorized
# operations. renamed resampler and resampler_kwargs in all results classes
# to best_resampler and best_resampler_params for consistency. rewrote
# _raw__repr__ to simply use all the __init__ arguments for representation.
# fix _raw__repr__ implementation to correctly use shorthand substitutions
# for cv_results, best_estimator, and best_resampler args.
#
# 06-25-2020
#
# more docstring cleanup for shizukaBaseCV and started cleaning up docstrings
# for shizukaSearchCV. swapped the names of shizukaBaseCV and shizukaCoreCV,
# which means that all functions using shizukaBaseCV need to be rewritten.
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
# using utils.is_method. change NoneType to None in docstrings. corrected :doc:
# references; turns out the reference is based on where the final HTML document
# generated by sphinx is located, not where the source is.
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

class BaseCVResults(FrozenClass):
    """Base class that all ``shizuka`` CV results classes inherit from.

    Do not use directly.

    .. note::

       A class instance is not frozen by default. The :meth:`_freeze` method
       must be manually called by the :meth:`__init__` method of any 
       :class:`BaseCVResults` subclass to "freeze" the class.

    Defines several common properties shared by all the subclasses. Note that
    the :attr:`cv_results` property is a dict and that subclasses may have
    :attr:`cv_results` with differing keys.

    :param best_estimator: Best fitted compatible estimator by validation score.
        Please see :doc:`../model_compat` for details on how the term 
        "compatible" is being used in this context.
    :type best_estimator: object
    :param best_score: Validation score of ``best_estimator``
    :type best_score: float
    :param scorer_name: Name of the metric used for scoring
    :type scorer_name: str
    :param cv_results: Per-fold validation results from cross-validation.
    :type cv_results: dict
    :param cv_iter: Number of validation folds/iterations.
    :type cv_iter: int
    :param total_time: Total runtime of cross-validation routine in seconds
    :type total_time: int
    :param shuffle: ``True`` if data was shuffled before being split into
        training and validation folds, ``False`` otherwise.
    :type shuffle: bool
    :param random_state: Seed used for k-fold data splitting. If no seed was
        provided, then defaults to ``None``.
    :type random_state: int or None
    :param best_resampler: The best class instance implementing
        :meth:`fit_resample` and :meth:`get_params` in the vein of ``imblearn``,
        or a custom resampling function. Default ``None``.

        .. note::

           Elaborate more on what "custom resampling function" means and
           give an explicit call signature for :meth:`fit_resample` and 
           :meth:`get_params`.

    :type best_resampler: object or function, optional
    :param best_resampler_params: The best keyword arguments passed to
        ``best_resampler``. Ignored if  ``best_resampler`` is ``None`` or if
        ``best_resampler`` implements :meth:`get_params` and
        :meth:`fit_resample` like the classes in ``imblearn`` package. Default
        ``None``.
    :type best_resampler_params: dict, optional
    """
    def __init__(self, best_estimator, best_score, scorer_name, cv_results,
                 cv_iter, total_time, shuffle, random_state,
                 best_resampler = None, best_resampler_params = None):
        self._best_estimator = best_estimator
        self._best_score = best_score
        self._scorer_name = scorer_name
        self._cv_results = cv_results
        # get parameters from best_estimator
        self._best_params = self.best_estimator.get_params()
        self._cv_iter = cv_iter
        self._total_time = total_time
        self._shuffle = shuffle
        self._random_state = random_state
        self._best_resampler = best_resampler
        # if best_resampler is None, set best_resampler_params to None
        if self._best_resampler is None: self._best_resampler_params = None
        # else if best_resampler implements fit_resample and get_params, call
        # get_params method to retrieve the resampler's parameters. we also
        # explicitly check that the methods are instance methods using
        # utils.is_method, which uses type() to check the state of the function.
        elif hasattr(self._best_resampler, "fit_resample") and \
             is_method(self._best_resampler.fit_resample) and \
             hasattr(self._best_resampler, "get_params") and \
             is_method(self._best_resampler.get_params):
            self._best_resampler_params = self._best_resampler.get_params()
        # else if best_resampler is a resampling function (should have two
        # unnamed with no defaults; optional keyword args allowed)
        elif hasattr(self._best_resampler, "__call__"):
            self._best_resampler_params = best_resampler_params
        else:
            raise TypeError("{0}: best_resampler must be class instance "
                            "implementing fit_resample and get_params, None, "
                            "or a function".format(self.__init__.__name__))

    def _raw__repr__(self):
        """Raw textual representation of the :class:`BaseCVResults`.

        Gives a string representation of a :class:`BaseCVResults` instance.
        Output is scikit-learn style, i.e.

        .. code:: python

           BaseCVResults(best_estimator=SVC, best_params={'C': 1.0, ... )

        This is the raw, unwrapped representation for the class instance. Each
        of the keyword arguments corresponds to an :meth:`__init__` parameter
        and its associated value.

        .. note::

           :meth:`_raw__repr__` is kept separate from the actual 
           :meth:`__repr__` method in case I change my mind on how subclasses
           should be represented.

        .. note::

           For brevity, the values for :attr:`best_estimator` and 
           :attr:`best_resampler` are substituted with the function/class
           instance name and the value of :attr:`cv_results` is substituted
           with ``"..."``

        :returns: Unwrapped string representation of the
            :class:`BaseCVResults` instance
        :rtype: str
        """
        # get name of the best estimator using name attribute of class type
        best_est_name = self._best_estimator.__class__.__name__
        # get resampler name; note that if resampler is a class instance, it is
        # not callable (while the function is callable).
        best_rs_name = None
        if self._best_resampler is None: best_rs_name = "None"
        elif hasattr(self._best_resampler, "__call__"):
            best_rs_name = self._best_resampler.__name__
        else: best_rs_name = self._best_resampler.__class__.__name__
        # use getfullargspec to get all __init__ arguments; drop self arg
        init_args = getfullargspec(self.__init__).args
        init_args.pop(0)
        # build the output string by collecting private attributes
        out_str = self.__class__.__name__ + "("
        for arg in init_args:
            # special substitutions for brevity
            if arg == "best_estimator": str_val = best_est_name
            elif arg == "cv_results": str_val = "..."
            elif arg == "best_resampler": str_val = best_rs_name
            # else use the repr() value
            else: str_val = repr(getattr(self, "_" + arg))
            out_str = out_str + arg + "=" + str_val + ", "
        # remove last ", ", replace with ")" and return
        return out_str[:-2] + ")"
        
    def __repr__(self):
        """Textual representation of the :class:`BaseCVResults`.
        
        Gives a wrapped string representation of a :class:`BaseCVResults`
        instance, essentially just wrapping the output of :meth:`_raw__repr__`
        with :func:`textwrap.fill`.

        :returns: Wrapped string representation of the :class:`BaseCVResults`
            instance
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
        """The best fitted compatible estimator, by validation score.

        See :doc:`../model_compat` for details on what "compatible" means in
        this context.

        :rtype: object
        """
        return self._best_estimator

    @property
    def best_score(self):
        """The score of the best fitted compatible estimator.

        See :attr:`scorer_name` for the scoring metric.

        :rtype: float
        """
        return self._best_score

    @property
    def scorer_name(self):
        """The name of the estimator scoring metric.

        .. note::

           :attr:`scorer_name` will have the value ``"default_score"`` if
           :attr:`best_estimator` is quasi-scikit-learn compatible and if the
           default scoring method, i.e. :meth:`score`, was used to score
           :attr:`best_estimator`. Typically ``"default_score"`` corresponds
           to :math:`R^2` for regression models and accuracy for classification
           models.

        :rtype: str
        """
        return self._scorer_name

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
            if data was shuffled before cross-validation splitting, ``False``
            if no data shuffling.
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
    def best_resampler(self):
        """The best resampler used with :attr:`best_estimator`.

        Is ``None`` if the best average validation performance was with no 
        resampler.

        :rtype: object or function
        """
        return self._resampler

    @property
    def best_resampler_params(self):
        """Best keyword parameters passed to the :attr:`best_resampler`.

        :returns: A dict of keyword arguments passed to :attr:`best_resampler`
            used with the best performing model. Is ``None`` if
            :attr:`best_resampler` is ``None`` or if the best resampler was
            most effective with default keyword arguments.
        :rtype: dict
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

        Each key corresponds to a :class:`numpy.ndarray` with shape 
        ``(cv_iter,)``, where each value at index ``i``, using zero-indexing,
        corresponds to a result for the ``(i + 1)``\ th cross-validation
        iteration.

        :rtype: dict
        """
        return self._cv_results
    

class CoreCVResults(BaseCVResults):
    """Core class holding model results from k-fold cross-validation routines.

    Contains results for a model trained with k-fold cross-validation and
    optional resampling. Inherits all the properties of its parent class,
    :class:`BaseCVResults`.

    .. note::

       Manual type checking is required before creating an instance of this
       class, but users should never need to create a :class:`BaseCVResults`
       instance themselves.

    :param best_estimator: Best fitted compatible estimator in terms of
        validation score. The definition of "compatible" in the context of this
        package can be found in :doc:`../model_compat`.
    :type best_estimator: object
    :param best_score: Validation score of ``best_estimator``
    :type best_score: float
    :param scorer_name: Name of the metric used for scoring
    :type scorer_name: str
    :param cv_results: A dict of per-fold cross-validation results. Each key is
        points to a :class:`numpy.ndarray` with shape ``(cv_iter,)``, i.e. the
        number of cross-validation folds used by the routine that generated the
        results. The keys and the descriptions of the data each key is
        associated with is described below.

        ``train_scores``
            A :class:`numpy.ndarray` of per-fold estimator training scores.

        ``cv_scores``
            A :class:`numpy.ndarray` of per-fold estimator validation scores.

        ``train_times``
            A :class:`numpy.ndarray` of per-fold estimator training times in
            seconds.

        ``resampling_times``
            A :class:`numpy.ndarray` of per-fold times in seconds needed for
            resampling.

        ``resampled_shapes``
            A :class:`numpy.ndarray` of per-fold resampled training data shapes,
            where each ``i``\ th fold data shape is 
            ``(n_rs_samples_i, n_features)``. If ``resampler`` is ``None``, then
            ``cv_results["resampled_shapes"]`` is ``None``.

        ``train_shapes``
            A :class:`numpy.ndarray` of per-fold training data shapes, where
            each ``i``\ the fold data shape is
            ``(n_train_samples_i, n_features)``.

        ``cv_shapes``
            A :class:`numpy.ndarray` of per-fold validation data shapes, where
            each ``i``\ th fold data shape is ``(n_val_samples_i, n_features)``.

    :type cv_results: dict
    :param cv_iter: Number of validation/folds iterations
    :type cv_iter: int
    :param total_time: Total runtime of cross-validation routine in seconds
    :type total_time: int
    :param shuffle: ``True`` if data was shuffled before being split into
        training and validation folds, ``False`` otherwise.
    :type shuffle: bool
    :param random_state: Seed used for k-fold data splitting. If no seed was 
        provided, then defaults to ``None``.
    :type random_state: int or None
    :param best_resampler: The best class instance implementing
        :meth:`fit_resample` and :meth:`get_params` in the vein of ``imblearn``,
        or a custom resampling function. Default ``None``.

        .. note::

           See the docstring for :class:`CoreCVResults` more for details.

    :type best_resampler: object or function, optional
    :param best_resampler_params: The best keyword arguments passed to
        ``best_resampler``. Ignored if  ``best_resampler`` is ``None`` or if
        ``best_resampler`` implements :meth:`get_params` and
        :meth:`fit_resample` like the classes in ``imblearn`` package. Default
        ``None``.
    :type best_resampler_params: dict, optional
    """
    def __init__(self, best_estimator, best_score, scorer_name, cv_results,
                 cv_iter, total_time, shuffle, random_state,
                 best_resampler = None, best_resampler_params = None):
        super().__init__(best_estimator, best_score, scorer_name, cv_results,
                         cv_iter, total_time, shuffle, random_state,
                         best_resampler = best_resampler,
                         best_resampler_params = best_resampler_params)
        # get best_cv_score, mean_cv_score, and std_cv_score from cv_results
        self._best_cv_score = max(self._cv_results["cv_scores"])
        self._mean_cv_score = mean(self._cv_results["cv_scores"])
        # note that standard deviation is calculated with n - 1 denominator here
        self._std_cv_score = std(self.cv_results["cv_scores"], ddof = 1)
        # freeze the class instance
        self._freeze()
        
    @property
    def best_cv_score(self):
        """The score of :attr:`best_estimator`, i.e. max of validation scores

        :rtype: float
        """
        return self._best_cv_score

    @property
    def mean_cv_score(self):
        """Average validation score among all the validation folds.

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

        :returns: A :class:`numpy.ndarray` of estimator training scores, shape
            ``(cv_iter,)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["train_scores"]
    
    @property
    def cv_scores(self):
        """Per-fold estimator validation scores.

        :returns: A :class:`numpy.ndarray` of estimator validation scores, shape
            ``(cv_iter,)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["cv_scores"]

    @property
    def train_times(self):
        """Per-fold estimator training times.

        :returns: A :class:`numpy.ndarray` of estimator training times in
            seconds, shape ``(cv_iter,)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["train_times"]

    @property
    def resampling_times(self):
        """Per-fold resampling times.

        :returns: A :class:`numpy.ndarray` of resampling times in seconds, shape
            ``(cv_iter,)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["resampling_times"]

    @property
    def train_shapes(self):
        """Per-fold training data shapes.

        :returns: A :class:`numpy.ndarray` of training data shapes, shape
            ``(cv_iter,)``, where the shape of the ``i``\ th fold training data 
            is ``(n_train_samples_i, n_features)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["train_shapes"]

    @property
    def resampled_shapes(self):
        """Per-fold training data shapes after resampling.

        :returns: A :class:`numpy.ndarray` of resampled training data shapes, 
            shape ``(cv_iter,)``, where the ``i``\ th fold resampled training data
            shape is ``(n_rs_samples_i, n_features)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["resampled_shapes"]

    @property
    def cv_shapes(self):
        """Per-fold validation data shapes.

        :returns: A :class:`numpy.ndarray` of validation data shapes, shape
            ``(cv_iter,)``, where the ``i``\ th fold validation data shape is
            ``(n_val_samples_i, n_features)``.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["cv_shapes"]

    @property
    def cv_results_df(self):
        """Returns :attr:`cv_results` as a :class:`pandas.DataFrame`.

        Syntactic sugar for returning the :attr:`cv_results` attribute as a 
        :class:`pandas.DataFrame`. Each row corresponds to a validation
        iteration (fold), so the :class:`pandas.DataFrame` will have
        :attr:`cv_iter` rows. The header names below corresponds to keys in
        :attr:`cv_results`.

        ::

            |--------------|-----------|-   -|------------------|-----------|
            | train_scores | cv_scores | ... | resampled_shapes | cv_shapes |
            |--------------|-----------|-   -|------------------|-----------|

        :rtype: :class:`pandas.DataFrame`
        """
        return DataFrame(self._cv_results)


class SearchCVResults(BaseCVResults):
    """Class with model results from hyperparameter search routines.

    Returned from hyperparameter search routines, containing results for a model
    trained with k-fold cross-validation and optional resampling. The best
    model is defined as the model whose average validation score, the mean of
    each of its per-fold validation scores, is the highest.

    .. note::

       Users should not need to create :class:`SearchCVResults` instances
       themselves.

    :param best_estimator: Best fitted compatible estimator, by average
        cross-validation score. Contains the hyperparameters that were best
        performing on average given the best resampler given by
        ``best_resampler`` and the best keyword arguments given by
        ``resampler_kwargs``.
    :type best_estimator: object
    :param best_score: Validation score of ``best_estimator``
    :type best_score: float
    :param scorer_name: Name of the metric used for scoring
    :type scorer_name: str
    :param cv_results: A dict with validation results for each model trained. If
        there are ``M`` possible hyperparameter combinations, then ``M`` models
        will have been trained. For each of the models, each key in
        ``cv_results`` points to a :class:`numpy.ndarray` of values where the
        ``i``\ th index of each :class:`numpy.ndarray`, under zero-indexing,
        corresponds to the ``(i + 1)``\ th of the ``M`` models trained. Keys 
        and descriptions of their associated data is below.
        
        ``param_name``
            For any model hyperparameter ``name``, a :class:`numpy.ndarray` of
            the values that ``name`` took for each model. The number of these
            key-value pairs is the number of hyperparameters passed to the
            search routine.

        ``resampler``
            A :class:`numpy.ndarray` of strings representing the name of the
            resampling class instance or function used for resampling the
            model's training data.

        ``rs_param_name``
            For any resampler keyword argument ``name``, a
            :class:`numpy.ndarray` of the values that ``name`` took for each
            resampler used on a particular model's training data. May contain
            values of :class:`numpy.nan` to indicate that a particular keyword
            argument is not one that is accepted by a resampler, as some keyword
            arguments may be ``None``.

            .. note::

               If none of the resamplers take keyword arguments, it is possible
               that these key-value mappings will not be present in
               ``cv_results``.

        ``foldk_cv_score``
            A :class:`numpy.ndarray` of model validation scores for the ``k``\ 
            th validation fold, where ``k`` is in the range ``[1, cv_iter]``.

        ``mean_cv_score``
            A :class:`numpy.ndarray` of model validation score averages, i.e.
            the average of each model's ``cv_iter`` per-fold validation scores.

        ``std_cv_score``
            A :class:`numpy.ndarray` of model validation score sample standard
            deviations, computed using :func:`numpy.std` with ``ddof = 1``.

        ``rank_cv_score``
            A :class:`numpy.ndarray` of model rankings, by mean validation
            score. The highest rank model will have rank 1, the lowest rank
            model will have rank ``M``.

        ``foldk_train_score``
            A :class:`numpy.ndarray` of model training scores for the ``k``\ th
            validation fold, where ``k`` is in the range ``[1, cv_iter]``.

        ``mean_train_score``
            A :class:`numpy.ndarray` of model training score averages, i.e. the
            average of each model's ``cv_iter`` per-fold training scores.

        ``std_train_score``
            A :class:`numpy.ndarray` of model training score sample standard
            deviations, computed using :func:`numpy.std` with ``ddof = 1``.

        ``rank_train_score``
            A :class:`numpy.ndarray` of model rankings, by mean training score.
            The highest rank model will have rank 1, the lowest rank ``M``.

            .. warning::

               If you are trying to select the model with the best validation
               performance, look at the rankings due to ``rank_cv_score`` instead.
               A model's training performance is biased upwards.
    
        ``mean_train_time``
            A :class:`numpy.ndarray` of average model training times, in
            seconds.

        ``mean_rs_time``
            A :class:`numpy.ndarray` of average resampling times, in seconds.

        ``foldk_train_shape``
            A :class:`numpy.ndarray` of shapes for the training data used during
            the ``k``\ th validation fold, where ``k`` is in the range
            ``[1, cv_iter]``. The shape for the ``i``\ th model has the format
            ``(n_train_samples_i, n_features)``.

        ``foldk_rs_shape``
            A :class:`numpy.ndarray` of shapes for the resampled training data
            used during the ``k``\ th validation fold , where ``k`` is in the
            range ``[1, cv_iter]``. The shape for the ``i``\ th model has the
            format ``(n_rs_samples_i,  n_features)``.

        ``foldk_cv_shape``
           A :class:`numpy.ndarray` of shapes for the validation data used
           during the ``k``\ th validation fold, where ``k`` is in the range
           ``[1, cv_iter]``. The shape for the ``i``\ th model has the format
           ``(n_val_samples_i, n_features)``.

    :type cv_results: dict
    :param cv_iter: Number of validation folds/iterations
    :type cv_iter: int
    :param total_time: Total runtime of the cross-validated hyperparameter search
        routine used, in seconds
    :type total_time: int
    :param shuffle: ``True`` if data was shuffled before being split into
        training and validation folds, ``False`` otherwise.
    :type shuffle: bool
    :param random_state: Seed used for k-fold data splitting. If no seed was 
        provided, then defaults to ``None``.
    :type random_state: int or None
    :param best_resampler: Of all the resamplers tried, the best resampler. Must
        be a class instance with :meth:`fit_resample` and :meth:`get_params`
        methods like the resampling classes defined in ``imblearn``, or a custom
        resampling function. Default ``None``.
    :type best_resampler: object or function, optional
    :param best_resampler_params: Of all keyword argument combinations, the
        best combination passed to ``best_resampler``. Ignored and set to
        ``None`` if ``resampler`` is ``None``. Default ``None`.
    :type best_resampler_params: dict, optional
    """
    def __init__(self, best_estimator, best_score, scorer_name, cv_results,
                 cv_iter, total_time, shuffle, random_state,
                 best_resampler = None, best_resampler_params = None):
        # call super() with appropriate parameters
        super().__init__(best_estimator, best_score, scorer_name, cv_results,
                         cv_iter, total_time, shuffle, random_state,
                         best_resampler = best_resampler,
                         best_resampler_params = best_resampler_params)
        # get best mean cv score. note best_resampler and best_resampler_params
        # are simply decorators for returning the resampler and resampler_params
        # attributes of the BaseCVResults parent class.
        self._best_mean_cv_score = max(self._cv_results["mean_cv_score"])
        # freeze
        self._freeze()
        
    ## attribute decorators ##
    @property
    def best_mean_cv_score(self):
        """The average validation score of :attr:`best_estimator`.

        :rtype: float
        """
        return self._best_mean_cv_score

    @property
    def resamplers(self):
        """The resamplers used in the hyperparameter search routine.

        :returns: A :class:`numpy.ndarray` of string names of the resamplers used
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["resampler"]

    @property
    def mean_cv_scores(self):
        """Average model validation scores.

        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["mean_cv_score"]

    @property
    def std_cv_scores(self):
        """Sample standard deviations of validation scores for each model.

        Sample standard deviations computed using :func:`numpy.std` with
        ``ddof = 1``.

        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["std_cv_score"]

    @property
    def rank_cv_scores(self):
        """Model rankings by average validation score.

        :returns: A :class:`numpy.ndarray` of model rankings by average
            validation score, where each ranking is a natural number and the
            best rank is 1.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["rank_cv_score"]

    @property
    def rank_train_scores(self):
        """Model rankings by average training score.

        :returns: A :class:`numpy.ndarray` of model rankings by average
            training score, where each ranking is a natural number and the
            best rank is 1.
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["rank_train_score"]

    @property
    def mean_train_scores(self):
        """Average model training scores.

        :rtype: :class:`numpy.ndarray`
        """

        return self._cv_results["mean_train_score"]

    @property
    def std_train_scores(self):
        """Sample standard deviations of training scores for each model.

        Sample standard deviations computed using :func:`numpy.std` with
        ``ddof = 1``.        

        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["std_train_score"]

    @property
    def mean_train_times(self):
        """Average model training times, in seconds.

        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["mean_train_time"]
        
    @property
    def mean_rs_times(self):
        """Average model resampling times, in seconds.
        
        :rtype: :class:`numpy.ndarray`
        """
        return self._cv_results["mean_rs_time"]

    @property
    def cv_results_df(self):
        """Returns :attr:`cv_results` as a :class:`pandas.DataFrame`.

        Syntactic sugar for returning :attr:`cv_results` attribute as a 
        :class:`pandas.DataFrame` with ``M`` rows, where results for each model
        /parameter set combination are placed in each row. ``M`` depends on the
        hyperparameter search routine used: for example, if an exhaustive grid
        search were used with ``m`` hyperparameters, where the ``i``\ th
        hyperparameter is provided ``n_i`` values to search through, then it is
        clear that ``M = n_1 * ... n_m``. 

        Header names, as in the illustration below, correspond to keys in
        :attr:`cv_results`.

        ::

          |--------------|-------------|-   -|---------------|--------------|-
          | param_kernel | param_gamma | ... | mean_cv_score | std_cv_score | ...
          |--------------|-------------|-   -|---------------|--------------|-

        .. note::

           Depending on the number of cross-validation folds chosen and whether
           resampling was used or not, the number of columns in
           :attr:`cv_results_df`  may change.

        :rtype: :class:`pandas.DataFrame`
        """
        return DataFrame(self._cv_results)


if __name__ == "__main__":
    print("{0}: do not run module as script".format(__module__), file = stderr)
