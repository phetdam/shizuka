.. repository readme

   Changelog:

   06-23-2020

   changed to .rst instead of .md file. added install info.

   01-10-2020

   initial creation.

shizuka
=======

by Derek Huang

:last updated: 06-23-2020
:file created: 01-10-2020

Giving you peace of mind by simplifying the analysis of statistical model performance. Work in progress.

If I think this package is good enough for public use, I might put it up on PyPI. Stay tuned.

Installation
------------

Either use the ``setup.py`` script directly or the Makefile. For the ``setup.py`` script, run

.. code:: bash

   python setup.py install

If using the Makefile, then try

.. code:: bash

   make install

Then try importing a module from ``shizuka`` to make sure that everything works as intended.

Modules
-------

* ``base``: Base methods and classes.

* ``plotting``: Module containing plotting methods wrapping ``matplotlib`` and ``seaborn`` to simplify model analysis.

* ``model_selection``: Module containing methods for more specialized model selection methods. Intended as a complement to a more complete machine learning package such as scikit-learn.
