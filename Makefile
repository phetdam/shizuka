# for simple shizuka package setup/install.
#
# Changelog:
#
# 06-23-2020
#
# initial creation.

CC          = gcc
CFLAGS      =
PYTHON      = python
SETUP_FLAGS = 

.PHONY: clean dummy

dummy:
	@echo "Please specify a target to build."

# removes emacs autosave files and local build directory
clean:
	@rm -vf *~
	@rm -vrf ./build

# build shizuka package
build:
	@$(PYTHON) setup.py build $(SETUP_FLAGS)

# install in site-packages directory for importing. builds if necessary.
install: build
	@$(PYTHON) setup.py install
