# Makefile for generating shizuka package documentation
#
# Changelog:
#
# 06-24-2020
#
# initial creation. got rid of catchall, added BUILDOPTS and clean, html
# targets. html target currently listed as dummy; configuration not complete.

SPHINXBUILD   = sphinx-build
# options for sphinx-build
BUILDOPTS     =
SOURCEDIR     = source
BUILDDIR      = build

# phony targets (take html off phony list later?)
.PHONY: help clean html

# "make" without argument is like "make help" (due to being first target)
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# remove all emacs autosave files in the directory
clean:
	@rm -vf *~

# make html documentation
html:
	@$(SPHINXBUILD) -M html $(BUILDOPTS) "$(SOURCEDIR)" "$(BUILDDIR)"

# no catch-alls; those are dangerous
