# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
DEFAULT_LANG  = zh_CN
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXINTL    = sphinx-intl
SPHINXPROJ    = maxframe
SOURCEDIR     = source
BUILDDIR      = build

# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) -D language=$(DEFAULT_LANG) $(SOURCEDIR)
I18NSPHINXLANGS = -l $(DEFAULT_LANG)

# make maxframe code available for sphinx builder
export PYTHONPATH := ../core

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

gettext:
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	$(SPHINXINTL) update -p $(BUILDDIR)/locale $(I18NSPHINXLANGS)
	python $(SOURCEDIR)/norm_zh.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
