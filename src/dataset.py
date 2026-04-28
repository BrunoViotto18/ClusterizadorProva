from enum import Enum

import kagglehub
import pandas as pd
from pydantic import BaseModel

from settings import DatasetSettings


def download_dataset(settings: DatasetSettings) -> pd.DataFrame:
    _ = kagglehub.dataset_download(settings.kaggle_key, output_dir=settings.path)

    dataset = pd.read_csv(
        settings.file_path,
        header=None,
        delimiter=settings.delimiter,
        names=settings.column_names,
    )

    return dataset
