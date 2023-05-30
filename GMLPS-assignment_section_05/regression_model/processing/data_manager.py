import re
from pathlib import Path
from typing import Any, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def get_first_cabin(row: Any) -> Union[str, int]:
    """Extract only the first cabin if more than 1 are available per passenger"""
    try:
        return row.split()[0]
    except AttributeError:
        return np.nan


def get_title(passenger: str) -> str:
    """Extracts the title (Mr, Ms, etc) from the name variable"""
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


def prepare_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    # replace question marks by NaN values
    data = dataframe.replace("?", np.nan)

    # extract title from name
    data["title"] = data["name"].apply(get_title)

    # drop unnecessary variables
    data.drop(
        labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True
    )

    # Extract first cabin only if more than 1 are available per passenger
    data["cabin"] = data["cabin"].apply(get_first_cabin)

    # cast numerical variables as floats
    data["fare"] = data["fare"].astype("float")
    data["age"] = data["age"].astype("float")

    data["cabin"] = data["cabin"].apply(get_first_cabin)

    return data


def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    return pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))


def load_and_transform_dataset(*, file_name: str) -> pd.DataFrame:
    loaded_dataframe = load_raw_dataset(file_name=file_name)
    transformed = prepare_pipeline(dataframe=loaded_dataframe)

    # rename variables beginning with numbers to avoid syntax errors later
    # transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
