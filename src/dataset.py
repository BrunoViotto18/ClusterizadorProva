import os
import numpy as np

import kagglehub
import pandas as pd

from abc import abstractmethod
from settings import AppSettings, DatasetSettings
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from typing import override

from transformer import DatasetTransformer


class DatasetNormalizer:

    def __init__(
        self,
        settings: AppSettings,
        numeric_scaler: DatasetTransformer,
        categorical_encoder: DatasetTransformer,
        ordinal_encoder: DatasetTransformer,
    ):
        self._settings = settings
        self._numeric_scaler: DatasetTransformer = numeric_scaler
        self._categorical_encoder: DatasetTransformer = categorical_encoder
        self._ordinal_encoder: DatasetTransformer = ordinal_encoder

    @property
    def normalized_columns(self):
        settings = self._settings.dataset
        return [
            *settings.numeric_columns,
            *self._categorical_encoder.get_normalized_columns(
                np.array(settings.categorical_columns)
            ),
            *settings.ordinal_columns,
        ]

    @staticmethod
    def fit_save(settings: AppSettings, dataset: pd.DataFrame) -> DatasetNormalizer:
        os.makedirs(settings.clusterizer.path, exist_ok=True)

        numeric_scaler = DatasetTransformer.fit_save(
            os.path.join(settings.clusterizer.path, "numeric_scaler.pkl"),
            dataset[settings.dataset.numeric_columns],
            MinMaxScaler(),
        )

        categorical_encoder = DatasetTransformer.fit_save(
            os.path.join(settings.clusterizer.path, "categorical_encoder.pkl"),
            dataset[settings.dataset.categorical_columns],
            OneHotEncoder(),
        )

        ordinal_encoder = DatasetTransformer.fit_save(
            os.path.join(settings.clusterizer.path, "ordinal_encoder.pkl"),
            dataset[settings.dataset.ordinal_columns],
            OrdinalEncoder(),
        )

        return DatasetNormalizer(
            settings, numeric_scaler, categorical_encoder, ordinal_encoder
        )

    @staticmethod
    def load(settings: AppSettings) -> DatasetNormalizer:
        numeric_scaler = DatasetTransformer.load(
            os.path.join(settings.clusterizer.path, "numeric_scaler.pkl")
        )

        categorical_encoder = DatasetTransformer.load(
            os.path.join(settings.clusterizer.path, "categorical_encoder.pkl")
        )

        ordinal_encoder = DatasetTransformer.load(
            os.path.join(settings.clusterizer.path, "ordinal_encoder.pkl")
        )

        return DatasetNormalizer(
            settings, numeric_scaler, categorical_encoder, ordinal_encoder
        )

    def normalize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        numeric_data_transformed = self._numeric_scaler.transform(
            dataset[self._settings.dataset.numeric_columns]
        )

        categorical_data_transformed = self._categorical_encoder.transform(
            dataset[self._settings.dataset.categorical_columns]
        )

        ordinal_data_transformed = self._ordinal_encoder.transform(
            dataset[self._settings.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_transformed.join(
            categorical_data_transformed
        ).join(ordinal_data_transformed)

        return dataset_transformed

    def denormalize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        numeric_data_inverse_transformed = self._numeric_scaler.inverse_transform(
            dataset[self._settings.dataset.numeric_columns]
        )

        categorical_data_inverse_transformed = (
            self._categorical_encoder.inverse_transform(
                dataset, np.array(self._settings.dataset.categorical_columns)
            )
        )

        ordinal_data_inverse_transformed = self._ordinal_encoder.inverse_transform(
            dataset[self._settings.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_inverse_transformed.join(
            categorical_data_inverse_transformed
        ).join(ordinal_data_inverse_transformed)

        dataset_transformed = dataset_transformed[self._settings.dataset.columns]

        return dataset_transformed


def download_dataset(settings: DatasetSettings) -> pd.DataFrame:
    _ = kagglehub.dataset_download(settings.kaggle_key, output_dir=settings.path)

    dataset = pd.read_csv(
        settings.file_path,
        header=None,
        delimiter=settings.delimiter,
        names=settings.columns,
    )

    return dataset
