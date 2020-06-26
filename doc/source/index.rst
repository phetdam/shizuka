.. shizuka documentation master file, created by
   sphinx-quickstart on Wed Jun 24 05:01:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

   Changelog:

   06-25-2020

   added small intro blurb to the document + note about dependency list changes.
   added model_selection module to autosummary; time for more warnings.

   06-24-2020

   initial creation. added cringey quote and link to intro.rst, model_compat.rst
   + field to toctree so that only document titles will be displayed. corrected
   improperly formatted footnote. added autosummary with toctree. change header
   for indices and tables to be "-" instead of "=" to reduce size.

   todo: configure autosummary and autodoc. 

Welcome to shizuka's documentation!
===================================

*Only those with a quiet heart can concentrate on the sound of the truth.* [#]_

A package for simplifying the analysis of statistical model performance, providing an abstracting layer over the standard Python scientific computing stack. 

.. note::

   Package dependencies subject to change without notice since package is in active development.

.. toctree::
   :maxdepth: 2
   :caption: Package contents
   :titlesonly:

   Introduction <intro>
   Estimator compatibility <model_compat>

.. note that ./modules will contain all the stub files generaetd by autosummary

   todo: make sure all the docstrings are of the right format
   
.. autosummary::
   :toctree: modules

   shizuka.base
   shizuka.model_selection
   shizuka.utils

.. [#] Yes, I really did just make this up. Not kidding.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
