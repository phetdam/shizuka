# Makefile for generating shizuka package documentation
#
# Changelog:
#
# 06-24-2020
#
# initial creation. got rid of catchall, added BUILDOPTS and clean, html
# targets. html target currently listed as dummy. changed clean target to
# clean up autosummary-generated files in $(SOURCEDIR)/modules to force
# clean rebuild of module documentation with `make html`.

SPHINXBUILD   = sphinx-build
# options for sphinx-build
BUILDOPTS     =
SOURCEDIR     = source
BUILDDIR      = build

# phony targets (take html off phony list later?)
.PHONY: help clean html

# "make" without argument is like "make help" (due to being first target)
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

# remove autosummary generated module files to force rebuild. also removes any
# emacs autosave files lingering around in this directory.
clean:
	@rm -vf $(SOURCEDIR)/modules/*.rst
	@rm -vf *~

# make html documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(BUILDOPTS)

# no catch-alls; those are dangerous
