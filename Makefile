PYTHON ?= python3
CUE ?= schema_template/.tools/cue/$(shell cat .cue-version)/cue
TASKS ?= $(wildcard schema_template/examples/*/)
CUE_SOURCES := schema_template/schema.cue schema_template/tools/validate_task.cue cue.mod/module.cue
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: setup test fmt fmt-check validate check
setup:
	$(PYTHON) schema_template/tools/install_cue.py

test:
	$(PYTHON) -m unittest discover -s schema_template/tests -v

fmt:
	$(CUE) fmt --files $(CUE_SOURCES)

fmt-check:
	$(CUE) fmt --check --files $(CUE_SOURCES)

validate:
	$(PYTHON) schema_template/tools/schema.py validate $(TASKS)

check: fmt-check test validate
