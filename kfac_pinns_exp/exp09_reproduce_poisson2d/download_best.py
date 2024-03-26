"""Download best runs of the random search from wandb."""

from os import path
from typing import Tuple

from pandas import DataFrame, concat, read_csv
from wandb import Api


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
