.. document explaining what "scikit-learn compatible" means.

   Changelog:

   06-24-2020

   initial creation. renamed to model_compat.rst. added intro and section
   headings for compatibility and scikit-learn compatibility.

   todo: write more details about normal and scikit-learn compatibility

Estimator compatibility
=======================

Despite efforts being made to keep the API flexible so that it may work with different types of models, there is a limit on what "kinds" of models are allowed to interface with the methods in the package. We therefore need to define what makes a model *compatible* with the methods in the package, and we will introduce two notions of compatbility, both of which are relatively relaxed. First, we will introduce the very loose notion of a model being *compatible* in this context and then discuss the more restrictive but still relatively loose notion of a model being *scikit-learn compatible*.

Compatible models
-----------------

Todo.

Scikit-learn compatible models
------------------------------

When we label a supervised learning model as being "scikit-learn compatible", we mean that the model supports has an API that is similar to and supports some core features endemic to that of the scikit-learn models. For a model to fit our very loose definition of "scikit-learn compatible", it must be that

1. The model is representable as an instance of a concrete class
2. The model has four key instance methods, namely ``fit``, ``get_params``, ``predict``, and ``score``.
