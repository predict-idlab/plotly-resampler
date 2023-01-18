# How to contribute

First of all, thank you for considering contributing to `plotly-resampler`.<br>
It's people like you that will help make `plotly-resampler` a great toolkit. 🤝

As usual, contributions are managed through GitHub Issues and Pull Requests.

As usual, contributions are managed through GitHub Issues and Pull Requests.  
We invite you to use GitHub's [Issues](https://github.com/predict-idlab/plotly-resampler/issues) to report bugs, request features, or ask questions about the project. To ask use-specific questions, please use the [Discussions](https://github.com/predict-idlab/plotly-resampler/discussions) instead.

If you are new to GitHub, you can read more about how to contribute [here](https://docs.github.com/en/get-started/quickstart/contributing-to-projects).

## How to develop locally

*Note: this guide is tailored to developers using linux*

The following steps assume that your console is at the root folder of this repository.

### Create a new (poetry) Python environment

It is best practice to use a new Python environment when starting on a new project.

We describe two options; 

<details>
<summary><i>Advised option</i>: using poetry shell</summary>
For dependency management we use poetry (read more below).<br>
Hence, we advise to use poetry shell to create a Python environment for this project.

1. Install poetry: https://python-poetry.org/docs/#installation <br>
   (If necessary add poetry to the PATH)
2. Create & activate a new python environment: <code>poetry shell</code>

After the poetry shell command your python environment is activated.
</details>

<details>
<summary><i>Alternative option</i>: using python-venv</summary>
As alternative option, you can create a Python environment by using python-venv

1. Create a new Python environment: <code>python -m venv venv</code>
2. Activate this environment; <code>source venv/bin/activate</code>
</details>

Make sure that this environment is activated when developing (e.g., installing dependencies, running tests).


### Installing & building the dependencies

We use [`poetry`](https://python-poetry.org/) as dependency manager for this project. 
- The dependencies for installation & development are written in the [`pyproject.toml`](pyproject.toml) file (which is quite similar to a requirements.txt file). 
- To ensure that package versions are consistent with everyone who works on this project poetry uses a [`poetry.lock`](poetry.lock) file (read more [here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)).

To install the requirements
```sh
pip install poetry  # install poetry (if you do use the venv option)
poetry install      # install all the dependencies
poetry build        # build the underlying C code
```

<details>
   <summary>
      <b>How to resolve the following error when running build.py:</b><br>
      <code>Unable to build the "plotly_resampler.aggregation.algorithms.lttbc" C extension; will use the slower python fallback. <br>
      command 'x86_64-linux-gnu-gcc' failed: No such file or directory
      </code>
   </summary>

   To resolve this error we suggest to install some additional packages as no gcc (C compiler was found on your PC):
   ```sh
   sudo apt-get install build-essential libssl-dev libffi-dev python-dev
   ```

</details>

### Formatting the code

We use [`black`](https://github.com/psf/black) and [`isort`](https://github.com/PyCQA/isort) to format the code.

To format the code, run the following command (more details in the [`Makefile`](Makefile)):
```sh
make format
```

### Checking the linting

We use [`ruff`](https://github.com/charliermarsh/ruff) to check the linting.

To check the linting, run the following command (more details in the [`Makefile`](Makefile)):
```sh
make lint
```

### Running the tests (& code coverage)

You can run the tests with the following code (more details in the [`Makefile`](Makefile)):

```sh
make test
```

To get the selenium tests working you should have Google Chrome installed.

If you want to visually follow the selenium tests;
* change the `TESTING_LOCAL` variable in [`tests/conftest.py`](tests/conftest.py) to `True`

### Generating the docs

When you've added or updated a feature; it is always a good practice to alter the 
documentation and [changelog.md](CHANGELOG.md).

The current listing below gives you the provided steps to regenerate the documentation.

1. Make sure that your python env is active (e.g., by running `poetry shell`)
2. Navigate to `sphinx/docs` and run from that directory:
```bash
sphinx-autogen -o _autosummary && make clean html
```

---

Bonus points for contributions that include a performance analysis with a benchmark script and profiling output (please report on the GitHub issue).

