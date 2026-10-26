PYTHON ?= python3
CUE ?= schema_template/.tools/cue/$(shell cat .cue-version)/cue

.PHONY: setup test fmt fmt-check validate build check
setup:
	$(PYTHON) schema_template/tools/install_cue.py

test:
	$(PYTHON) -m unittest discover -s schema_template/tests -v

fmt:
	$(CUE) fmt --files schema_template cue.mod/module.cue

fmt-check:
	$(CUE) fmt --check --files schema_template cue.mod/module.cue

validate:
	$(PYTHON) schema_template/tools/schema.py validate --all

build:
	$(PYTHON) schema_template/tools/schema.py build --all

check: fmt-check test
	$(PYTHON) schema_template/tools/schema.py check --all
