black = black plotly_resampler examples tests
isort = isort plotly_resampler examples tests

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint
lint:
	poetry run ruff plotly_resampler tests
	poetry run $(isort) --check-only --df
	poetry run $(black) --check --diff

.PHONY: test
test:
	poetry run pytest --cov-report term-missing --cov=plotly_resampler tests

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f .coverage
	rm -rf build

	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '*.cpython-*' `
