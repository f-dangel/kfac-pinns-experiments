# Developer guide

This guide describes principles and workflows for developers.

## Setup

We recommend programming in a fresh virtual environment. You can set up the
`conda` environment and activate it

```bash
make conda-env
conda activate kfac_pinns_exp
```

If you don't use `conda`, set up your preferred environment and run

```bash
pip install -e ."[lint,test]"
```
to install the package in editable mode, along with all required development dependencies
(the quotes are for OS compatibility, see
[here](https://github.com/mu-editor/mu/issues/852#issuecomment-498759372)).

## Continuous integration

To standardize code style and enforce high quality, checks are carried out with
Github actions when you push. You can also run them locally, as they are managed
via `make`:

- Run tests with `make test`

- Run all linters with `make lint`, or separately with:

    - Run auto-formatting and import sorting with `make black` and `make isort`

    - Run linting with `make flake8`

    - Run docstring checks with `make pydocstyle-check` and `make darglint-check`

## Documentation

We use the [Google docstring
convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
and `mkdocs` which allows using markdown syntax in a docstring to achieve
formatting.

## Reproducibility
To reproduce the experiments shown in the [paper](https://arxiv.org/abs/2405.15603) consult the following table.

| Figure     | Group Plot Name                   | Uses Experiments                |
|------------|-----------------------------------|---------------------------------|
| Figure 1   | exp17_groupplot_poisson_2d        | exp09, exp15, exp20             |
| Figure 2   | exp30_heat4d_groupplot            | exp27, exp28, exp29             |
| Figure 3   | exp33_poisson_bayes_groupplot     | exp14, exp26, exp32             |
| Figure A4  | exp17_groupplot_poisson_2d        | exp09, exp15, exp20             | 
| Figure A5  | exp42_visualize_solutions         | exp09, exp13                    |
| Figure A6  | exp18_groupplot_poisson5d         |exp10, exp16, exp19              |
| Figure A7  | exp21_poisson_10d                 |                                 |
| Figure A8  | exp33_poisson_bayes_groupplot     | exp14, exp26, exp32             |
| Figure A9  | exp24_heat1d_groupplot            | exp13, exp22, exp23             |
| Figure A10 | exp42_visualize_solutions         | exp13, exp09                    |
| Figure A11 | exp30_heat4d_groupplot            | exp27, exp28, exp29             |
| Figure A12 | exp41_errorbars_exp28             | exp28                           |
| Figure A13 | exp31_heat4d_mlp_tanh_256_bayes   |                                 |