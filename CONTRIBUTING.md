# How to contribute

First of all, thank you for considering contributing to `plotly-resampler`.<br>
It's people like you that will help make `plotly-resampler` a great toolkit.

As usual, contributions are managed through GitHub Issues and Pull Requests.

We are welcoming contributions in the following forms:
* **Bug reports**: when filing an issue to report a bug, please use the search tool to ensure the bug hasn't been reported yet;
* **New feature suggestions**: if you think `plotly-resampler` should include a new feature, please open an issue to ask for it (of course, you should always check that the feature has not been asked for yet :). Think about linking to a pdf version of the paper that first proposed the method when suggesting a new algorithm. 
* **Bugfixes and new feature implementations**: if you feel you can fix a reported bug/implement a suggested feature yourself, do not hesitate to:
  1. fork the project;
  2. implement your bugfix;
  3. submit a pull request referencing the ID of the issue in which the bug was reported / the feature was suggested;
    
When submitting code, please think about code quality, adding proper docstrings, and including thorough unit tests with high code coverage.


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

We use [poetry](https://python-poetry.org/) as dependency manager for this project. 
- The dependencies for installation & development are written in the [pyproject.toml](pyproject.toml) file (which is quite similar to a requirements.txt file). 
- To ensure that package versions are consistent with everyone who works on this project poetry uses a [poetry.lock](poetry.lock) file (read more [here](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)).

To install the requirements
```sh
pip install poetry  # install poetry (if you do use the venv option)
poetry install  # install all the dependencies
poetry build  # build the underlying C code
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

### Running the tests (& code coverage)

You can run the test with the following code:

```sh
poetry run pytest --cov-report term-missing --cov=plotly_resampler tests
```

To get the selenium tests working you should have Google Chrome installed.

If you want to visually follow the selenium tests;
* change the `TESTING_LOCAL` variable in [tests/conftest.py](tests/conftest.py) to `True`

### Generating the docs

When you've added or updated a feature; it is always a good practice to alter the 
documentation and [changelog.md](CHANGELOG.md).

The current listing below gives you the provided steps to regenerate the documentation.

1. Make sure that your python env is active (e.g., by running `poetry shell`)
2. Navigate to `sphinx/docs` and run from that directory:
```bash
sphinx-autogen -o _autosummary && make clean html
```

## More details on Pull requests

The preferred workflow for contributing to plotly-resampler is to fork the
[main repository](https://github.com/predict-idlab/plotly-resampler) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/predict-idlab/plotly-resampler)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the plotly-resampler repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/plotly-resampler.git
   $ cd plotly-resampler
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

### Pull Request Checklist

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. 
   This will make sure a link back to the original issue is created.

-  All public methods should have informative *numpy* docstrings with sample
   usage presented as doctests when appropriate. Validate whether the generated 
   documentation is properly formatted (see below). 

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example notebook in the ``plotly-resampler/examples/`` folder or add the functionality in an 
   existing notebook. Have a look at other examples for reference. 
   Examples should demonstrate why the new functionality is useful in practice and, 
   if possible, benchmark/integrate with other packages.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in master and pass for the PR code.


---

Bonus points for contributions that include a performance analysis with a benchmark script and profiling output (please report on the GitHub issue).

