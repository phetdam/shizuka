# setup.py for shizuka package
#
# Changelog:
#
# 06-23-2020
#
# initial creation. maybe i should use setuptools instead?

from distutils.core import setup

def _setup():
    setup(name = "shizuka",
          version = "0.0.1",
          description = "Simplifying the analysis of statistical model performance",
          author = "Derek Huang",
          packages = ["shizuka"]
    )

if __name__ == "__main__":
    _setup()
