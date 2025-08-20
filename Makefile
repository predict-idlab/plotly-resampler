black = black plotly_resampler examples tests

.PHONY: format
format:
	$(black)
	poetry run ruff check plotly_resampler tests

.PHONY: lint
lint:
	poetry run ruff check plotly_resampler tests
	poetry run $(black) --check --diff

.PHONY: test
test:
	poetry run pytest --cov-report term-missing --cov=plotly_resampler tests

.PHONY: docs
docs:
	poetry run mkdocs build -c -f mkdocs.yml

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
