"""Utility functions for Weights & Biases."""

from copy import deepcopy
from glob import glob
from os import path, remove
from typing import Any, Dict, List, Tuple, Union

from pandas import DataFrame, read_csv
from wandb import Api
from wandb.apis.public.sweeps import Sweep


def show_sweeps(entity: str, project: str) -> List[Sweep]:
    """Print ids and names of sweeps in a project.

    This is useful to map sweep ids to human-readable names.

    Args:
        entity: The team name on wandb.
        project: The name from the 'Projects' tab on wandb.

    Returns:
        A list of sweep objects.
    """
    api = Api()
    sweeps = api.project(project, entity=entity).sweeps()
    print(f"Found the following sweeps in {entity}/{project}:")
    for sweep in sweeps:
        print(f"\tid: {sweep.id}, ({sweep.config['name']})")
    return list(sweeps)


def load_best_run(
    entity: str,
    project: str,
    sweep_id: str,
    save: bool = True,
    savedir: str = ".",
    update: bool = True,
) -> Tuple[DataFrame, DataFrame]:
    """Load history and meta-data for the best run from wandb.

    Args:
        entity: The team name on wandb.
        project: The name from the 'Projects' tab on wandb.
        sweep_id: The id of the sweep from the 'Sweeps' tab on wandb.
        save: Whether to save history and data locally to csv. Default is `True`.
        savedir: The directory to save the csv files. Default is the current directory.
        update: Whether to request the best run from wandb. If `False`, tries loading
            from an existing local file. Default is `True`.

    Returns:
        The history and meta-data data frames of the best run.
    """
    prefix = path.abspath(path.join(savedir, f"{entity}_{project}_{sweep_id}_best"))
    history_path = f"{prefix}_history.csv"
    meta_path = f"{prefix}_meta.csv"

    # try loading from local files
    if path.exists(history_path) and path.exists(meta_path) and not update:
        print(f"Loading from previous download:\n\t{history_path}\n\t{meta_path}")
        return read_csv(history_path), read_csv(meta_path)

    # determine the best run
    sweep = Api().sweep(f"{entity}/{project}/{sweep_id}")
    run = sweep.best_run()

    # extract logged quantities
    config = {k: v for k, v in run.config.items() if not k.startswith("_")}
    df_meta = DataFrame({"config": [config], "name": [run.name]})
    df_history = run.history()

    if save:
        print(f"Saving downloaded files locally:\n\t{history_path}\n\t{meta_path}")
        df_history.to_csv(history_path, index=False)
        df_meta.to_csv(meta_path, index=False)

    return df_history, df_meta


def remove_unused_runs(keep: List[str], best_run_dir: str = ".", verbose: bool = True):
    """Remove all saved best runs in a directory.

    Args:
        keep: The list of sweep ids to keep.
        best_run_dir: The directory where the best runs are saved.
            Default is the current directory.
        verbose: Whether to print the files being removed. Default is `True`.
    """
    csv_files = glob(path.join(best_run_dir, "*.csv"))
    delete = [csv for csv in csv_files if all(sweep_id not in csv for sweep_id in keep)]
    for f in delete:
        if verbose:
            print(f"Removing unused sweep data in {f}.")
        remove(f)


class WandbRunFormatter:
    """Class to format command line args of a wandb run to LaTeX."""

    HYPERPARAMETERS = {
        "SGD": {"SGD_lr": "learning rate", "SGD_momentum": "momentum"},
        "Adam": {"Adam_lr": "learning rate"},
        "HessianFree": {
            "HessianFree_curvature_opt": "curvature matrix",
            "HessianFree_damping": "initial damping",
            "HessianFree_no_adapt_damping": "constant damping",
            "HessianFree_cg_max_iter": "maximum CG iterations",
        },
        "LBFGS": {
            "LBFGS_lr": "learning rate",
            "LBFGS_history_size": "history size",
        },
        "ENGD": {
            "ENGD_damping": "damping",
            "ENGD_ema_factor": "exponential moving average",
            "ENGD_initialize_to_identity": "initialize Gramian to identity",
        },
        "KFAC": {
            "KFAC_damping": "damping",
            "KFAC_momentum": "momentum",
            "KFAC_ema_factor": "exponential moving average",
            "KFAC_initialize_to_identity": "initialize Kronecker factors to identity",
        },
    }

    @classmethod
    def num(cls, value: Union[float, int, bool, str]) -> str:
        """Format a value to LaTeX.

        Args:
            value: The value to format.

        Returns:
            The LaTeX-formatted string.
        """
        # Strangely, some numbers are stored as float in wandb. I noticed that they
        # follow the pattern 1e-X. FIX: Convert to float here explicitly
        if isinstance(value, str) and value.startswith("1e-"):
            value = float(value)

        if isinstance(value, str):
            spellings = {"ggn": "GGN", "hessian": "Hessian"}
            return r"$\text{" + spellings.get(value, value) + "}$"
        elif isinstance(value, bool):
            return cls.num({True: "yes", False: "no"}[value])
        elif isinstance(value, float):
            value_str = f"{value:.6e}" if len(str(value)) > 10 else str(value)
            return r"$\num[scientific-notation=true]{" + value_str + "}$"
        elif isinstance(value, int):
            return r"$\num[scientific-notation=false]{" + str(value) + "}$"
        else:
            return str(value)

    @classmethod
    def to_tex(cls, directory: str, args: Dict[str, Any]):
        """Format a dictionary of command line args to LaTeX and save to tex file.

        Args:
            directory: The directory to save the tex file.
            args: The dictionary of command line args.
        """
        optimizer = args["optimizer"]
        hyperparams = cls.HYPERPARAMETERS[optimizer]

        if optimizer == "ENGD":
            approximation = args["ENGD_approximation"]
            optimizer = {
                "full": "ENGD_full",
                "per_layer": "ENGD_per_layer",
                "diagonal": "ENGD_diagonal",
            }[approximation]
        elif optimizer == "KFAC" and args["KFAC_lr"] == "auto":
            optimizer = "KFAC_auto"
            hyperparams = {k: v for k, v in hyperparams.items() if k != "KFAC_momentum"}

        # create human-readable description
        text = ", ".join(
            [f"{value}: {cls.num(args[key])}" for key, value in hyperparams.items()]
        )

        tex_file = path.join(directory, f"best_{optimizer}.tex")
        with open(tex_file, "w") as f:
            f.write(text)


class WandbBayesianRunFormatter(WandbRunFormatter):
    """Class to format command line args of a Bayesian wandb run to LaTeX."""

    HYPERPARAMETERS = deepcopy(WandbRunFormatter.HYPERPARAMETERS)
    for params in HYPERPARAMETERS.values():
        params["N_Omega"] = r"$N_{\Omega}$"
        params["N_dOmega"] = r"$N_{\partial\Omega}$"
        params["batch_frequency"] = "batch sampling frequency"


class WandbSweepFormatter(WandbRunFormatter):
    """Class to format wandb sweeps into human-readable LaTeX files."""

    @classmethod
    def dist(cls, parameter: Dict) -> str:
        """Format a parameter entry of a sweep into LaTeX.

        Args:
            parameter: The parameter entry of a sweep.

        Returns:
            The LaTeX-formatted string describing the parameter.

        Raises:
            NotImplementedError: If the distribution is not recognized.
        """
        if "distribution" not in parameter:
            return cls.num(parameter["value"])

        dist = parameter["distribution"]
        if dist == "log_uniform_values":
            min_str = cls.num(parameter["min"]).replace("$", "")
            max_str = cls.num(parameter["max"]).replace("$", "")
            return r"$\mathcal{LU}([" + f"{min_str}; {max_str}])$"
        elif dist == "uniform":
            min_str = cls.num(parameter["min"]).replace("$", "")
            max_str = cls.num(parameter["max"]).replace("$", "")
            return r"$\mathcal{U}([" + f"{min_str}; {max_str}])$"
        elif dist == "categorical":
            values = ",".join(
                [cls.num(v).replace("$", "") for v in parameter["values"]]
            )
            return r"$\mathcal{C}(\{" + values + r"\})$"
        elif dist == "int_uniform":
            values = ",".join(
                [
                    cls.num(v).replace("$", "")
                    for v in [
                        parameter["min"],
                        parameter["min"] + 1,
                        r"$\dots$",
                        parameter["max"],
                    ]
                ]
            )
            return r"$\mathcal{C}(\{" + values + r"\})$"

        else:
            raise NotImplementedError(f"Unknown distribution {dist}.")

    @classmethod
    def to_tex(cls, directory: str, args: Dict[str, Dict[str, Any]]):
        """Format a dictionary of command line args to LaTeX and save to tex file.

        Args:
            directory: The directory to save the tex file.
            args: The nested dictionary of sweep arguments.
        """
        # extract optimizer name
        cmd_args = args["command"]
        (optimizer,) = [
            arg.removeprefix("--optimizer=")
            for arg in cmd_args
            if "--optimizer=" in arg
        ]
        hyperparams = cls.HYPERPARAMETERS[optimizer]
        parameters = args["parameters"]

        # add suffix for KFAC and ENGD specifying the variant.
        if (
            optimizer == "KFAC"
            and "KFAC_lr" in parameters
            and parameters["KFAC_lr"].get("value") == "auto"
        ):
            optimizer = "KFAC_auto"
            hyperparams = {k: v for k, v in hyperparams.items() if k != "KFAC_momentum"}
        if optimizer == "ENGD":
            optimizer = {
                "full": "ENGD_full",
                "per_layer": "ENGD_per_layer",
                "diagonal": "ENGD_diagonal",
            }[parameters["ENGD_approximation"]["value"]]

        # create human-readable description
        text = ", ".join(
            [
                f"{value}: {cls.dist(parameters[key])}"
                for key, value in hyperparams.items()
            ]
        )

        tex_file = path.join(directory, f"sweep_{optimizer}.tex")
        with open(tex_file, "w") as f:
            f.write(text)


class WandbBayesianSweepFormatter(WandbSweepFormatter):
    """Class to format Bayesian sweeps into human-readable LaTeX."""

    HYPERPARAMETERS = deepcopy(WandbSweepFormatter.HYPERPARAMETERS)
    for params in HYPERPARAMETERS.values():
        params["N_Omega"] = r"$N_{\Omega}$"
        params["N_dOmega"] = r"$N_{\partial\Omega}$"
        params["batch_frequency"] = "batch sampling frequency"
