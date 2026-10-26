PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

ifeq ($(strip $(PYTHON)),)
$(error Python was not found; set PYTHON=/path/to/python)
endif

.PHONY: clean smoke test

smoke:
	$(PYTHON) test_suite.py smoke

test:
	$(PYTHON) test_suite.py test

clean:
	rm -rf test-results __pycache__ algorithm/__pycache__ oracle/__pycache__
