# setup.py for shizuka package
#
# Changelog:
#
# 06-25-2020
#
# added license and install_requires to _setup. make long description read from
# README.rst and install_requires read from requirements.json.
#
# 06-23-2020
#
# initial creation. maybe i should use setuptools instead?

import json
from setuptools import setup

def _setup():
    # short and long descriptions
    desc_short = "Simplifying the analysis of statistical model performance"
    with open("README.rst", "r") as rmf:
        desc_long = rmf.read()
    # get requirements from requirements.json
    with open("requirements.json", "r") as rqf:
        reqs = json.load(rqf)
    # setup
    setup(name = "shizuka",
          version = "0.0.1",
          description = desc_short,
          long_description = desc_long,
          long_description_content_type = "text/x-rst",
          author = "Derek Huang",
          packages = ["shizuka"],
          license = "MIT",
          install_requires = reqs
    )

if __name__ == "__main__":
    _setup()
