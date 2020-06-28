.. document explaining what "quasi-scikit-learn compatible" means.

   Changelog:

   06-28-2020

   edited wording for opening paragraph. changed inline literals for method
   names to using :meth: tag and added examples for defining get_params as an
   instance method of a compatible model. finished compatible model section.

   06-24-2020

   initial creation. renamed to model_compat.rst. added intro and section
   headings for compatibility and quasi-scikit-learn compatibility. Changed
   statement of compatibility to include the term "quasi" since not all the
   scikit-learn features are supported. added function signatures.

   todo: write more details about normal and quasi-scikit-learn compatibility

Estimator compatibility
=======================

Although the API is flexible enough to work with different types of models, there is a limit on what "kinds" of models are allowed to interface with the methods in the package. We therefore need to define what makes a model *compatible* with the methods in the package, and we will introduce two notions of compatbility, both of which are relatively relaxed. We will introduce the very loose notion of a model being *compatible* in this context and then discuss the more restrictive but still relatively loose notion of a model being *quasi-scikit-learn compatible*.

Compatible models
-----------------

A *compatible* model is a class that implements the instance method :meth:`get_params`, which returns the model's hyperparameters. With type annotations, :meth:`get_params` is defined as

.. code:: python

   def get_params(self) -> dict:
       # ... code ...

The signature of the function implies that all hyperparameters must be keyword arguments, the same practice as is used in scikit-learn. Accordingly, the simplest type of model that would be considered compatible would contain a few hyperparameters as instance variables, which :meth:`get_params` would return as a dict. An example is below.

.. code:: python

   class SimpleModel:

       def __init__(self, param1 = "default1", param2 = "default2"):
           self.param1 = param1
	   self.param2 = param2

       def get_params(self): return {"param1": param1, "param2": param2}

As in the example, it is encouraged that any user-defined model classes take zero positional arguments besides ``self`` in the vein of scikit-learn. However, the example is admittedly a good example of how **not** to define :meth:`get_params`. In the general case of ``n`` many hyperparameters, it obviously becomes messy to write all the dictionary keys by hand. A more general :meth:`get_params` example, using :func:`inspect.getfullargspec`, would be to do something like

.. code:: python

   def get_params(self):
       argnames = inspect.getfullargspec(self.__init__).args[1:]
       params = {}
       for a in argnames: params[a] = getattr(self, a)
       return params

The first line sets ``argnames`` to be a list of all the :meth:`__init__` method keyword argument names, which we may then iterate through and use :func:`getattr` to get the value of each respective keyword argument from the class instance. Note that the definition of :meth:`get_params` above only returns the keyword arguments and values of the :meth:`__init__` method and that a class may define other instance attributes that will not show up in the dict returned by :meth:`get_params`.

Fitting and predicting with the model can be done through external functions or through class instance methods, whichever are desired by the user.

Quasi-scikit-learn compatible models
------------------------------------

When we label a supervised learning model as being "quasi-scikit-learn compatible", we mean that the model supports has an API that is similar to and supports some core features endemic to that of the scikit-learn models. For a model to fit our very loose definition of "quasi-scikit-learn compatible", it must be that

1. The model is representable as an instance of a concrete class
2. The model has four key instance methods, namely :meth:`fit`, :meth:`get_params`, :meth:`predict`, and :meth:`score`.

Function signatures. Note that we allow keyword arguments.
   
.. code:: python

   fit(self: object, X: numpy.ndarray, y: numpy.ndarray, **kwargs) -> object
   get_params(self: object, **kwargs) -> dict
   predict(self: object, X: numpy.ndarray, **kwargs) -> numpy.ndarray
   score(self: object, X: numpy.ndarray, y: numpy.ndarray, **kwargs) -> float

Here ``X`` should have shape ``(n_samples, n_features)`` and ``y`` should have shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.
