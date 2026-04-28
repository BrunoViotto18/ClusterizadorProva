import os
import pickle
import numpy as np

import kagglehub
import pandas as pd

from abc import ABC, abstractmethod
from settings import AppSettings, DatasetSettings
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from typing import Any, override


class Transfomer(ABC):

    @staticmethod
    @abstractmethod
    def fit_save(path: str, dataset: pd.DataFrame, settings: AppSettings) -> Transfomer:
        pass

    @staticmethod
    @abstractmethod
    def load(path: str) -> Transfomer:
        pass

    @abstractmethod
    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def inverse_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform_column(self, column: str) -> list[str]:
        return [column]

    def transform_columns(self, columns: list[str]) -> list[str]:
        transformed_columns: list[str] = []

        for column in columns:
            transformed = self.transform_column(column)
            transformed_columns.extend(transformed)

        return transformed_columns


class MinMaxTransformer(Transfomer):
    def __init__(self, scaler: MinMaxScaler):
        self._scaler: MinMaxScaler = scaler

    @staticmethod
    @override
    def fit_save(
        path: str, dataset: pd.DataFrame, settings: AppSettings
    ) -> MinMaxTransformer:
        transformer = MinMaxScaler()

        transformer = transformer.fit(dataset)

        with open(path, "wb") as f:
            pickle.dump(transformer, f)

        return MinMaxTransformer(transformer)

    @staticmethod
    @override
    def load(path: str) -> MinMaxTransformer:
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        return MinMaxTransformer(scaler)

    @override
    def transform(self, dataset: pd.DataFrame):
        columns = dataset.columns

        transformed_data = self._scaler.transform(dataset)

        transformed_dataset = pd.DataFrame(transformed_data, columns=columns)

        return transformed_dataset

    @override
    def inverse_transform(self, dataset: pd.DataFrame):
        transformed_data = self._scaler.inverse_transform(dataset)

        transformed_dataset = pd.DataFrame(transformed_data, columns=dataset.columns)

        return transformed_dataset


class OneHotTransformer(Transfomer):
    def __init__(self, scaler: OneHotEncoder):
        self._scaler: OneHotEncoder = scaler

    @staticmethod
    @override
    def fit_save(
        path: str, dataset: pd.DataFrame, settings: AppSettings
    ) -> OneHotTransformer:
        transformer = OneHotEncoder()

        transformer = transformer.fit(dataset)

        with open(path, "wb") as f:
            pickle.dump(transformer, f)

        return OneHotTransformer(transformer)

    @staticmethod
    @override
    def load(path: str) -> OneHotTransformer:
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        return OneHotTransformer(scaler)

    @override
    def transform(self, dataset: pd.DataFrame):
        columns = dataset.columns

        transformed_data: Any = self._scaler.transform(dataset)

        transformed_data = transformed_data.toarray()

        columns = self._scaler.get_feature_names_out(columns)

        transformed_dataset = pd.DataFrame(transformed_data, columns=columns)

        return transformed_dataset

    @override
    def inverse_transform(self, dataset: pd.DataFrame):
        transformed_data = self._scaler.inverse_transform(dataset)

        columns = self._scaler.feature_names_in_

        transformed_dataset = pd.DataFrame(transformed_data, columns=columns)

        return transformed_dataset

    @override
    def transform_column(self, column: str) -> list[str]:
        return [*self._scaler.get_feature_names_out([column])]


class OrdinalTransformer(Transfomer):
    def __init__(self, scaler: OrdinalEncoder):
        self._scaler: OrdinalEncoder = scaler

    @staticmethod
    @override
    def fit_save(
        path: str, dataset: pd.DataFrame, settings: AppSettings
    ) -> OrdinalTransformer:
        transformer = OrdinalEncoder()

        transformer = transformer.fit(dataset)

        with open(path, "wb") as f:
            pickle.dump(transformer, f)

        return OrdinalTransformer(transformer)

    @staticmethod
    @override
    def load(path: str) -> OrdinalTransformer:
        with open(path, "rb") as f:
            scaler = pickle.load(f)

        return OrdinalTransformer(scaler)

    @override
    def transform(self, dataset: pd.DataFrame):
        columns = dataset.columns

        transformed_data = self._scaler.transform(dataset)

        transformed_dataset = pd.DataFrame(transformed_data, columns=columns)

        return transformed_dataset

    @override
    def inverse_transform(self, dataset: pd.DataFrame):
        transformed_data = self._scaler.inverse_transform(dataset)

        transformed_dataset = pd.DataFrame(transformed_data, columns=dataset.columns)

        return transformed_dataset


class DatasetTransformer(Transfomer):

    def __init__(
        self,
        minmax_transformer: MinMaxTransformer,
        onehot_transformer: OneHotTransformer,
        ordinal_transformer: OrdinalTransformer,
    ):
        self._minmax_transformer: MinMaxTransformer = minmax_transformer
        self._onehot_transformer: OneHotTransformer = onehot_transformer
        self._ordinal_transformer: OrdinalTransformer = ordinal_transformer

    @staticmethod
    def fit_save(
        path: str, dataset: pd.DataFrame, settings: AppSettings
    ) -> DatasetTransformer:

        parent_directory = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)

        minmax_transformer = MinMaxTransformer.fit_save(
            os.path.join(parent_directory, "minmax_transformer.pkl"),
            dataset[settings.dataset.numeric_columns],
            settings,
        )

        onehot_transformer = OneHotTransformer.fit_save(
            os.path.join(parent_directory, "onehot_transformer.pkl"),
            dataset[settings.dataset.categorical_columns],
            settings,
        )

        ordinal_transformer = OrdinalTransformer.fit_save(
            os.path.join(parent_directory, "ordinal_transformer.pkl"),
            dataset[settings.dataset.ordinal_columns],
            settings,
        )

        return DatasetTransformer(
            minmax_transformer, onehot_transformer, ordinal_transformer
        )

    @staticmethod
    def load(path: str) -> DatasetTransformer:
        parent_directory = os.path.dirname(path)

        minmax_transformer = MinMaxTransformer.load(
            os.path.join(parent_directory, "minmax_transformer.pkl")
        )

        onehot_transformer = OneHotTransformer.load(
            os.path.join(parent_directory, "onehot_transformer.pkl")
        )

        ordinal_transformer = OrdinalTransformer.load(
            os.path.join(parent_directory, "ordinal_transformer.pkl")
        )

        return DatasetTransformer(
            minmax_transformer, onehot_transformer, ordinal_transformer
        )

    def transform(self, dataset: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
        numeric_data_transformed = self._minmax_transformer.transform(
            dataset[settings.dataset.numeric_columns]
        )

        categorical_data_transformed = self._onehot_transformer.transform(
            dataset[settings.dataset.categorical_columns]
        )

        ordinal_data_transformed = self._ordinal_transformer.transform(
            dataset[settings.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_transformed.join(
            categorical_data_transformed
        ).join(ordinal_data_transformed)

        columns = self.transform_columns(settings.dataset.columns)
        dataset_transformed = dataset_transformed[columns]

        return dataset_transformed

    def inverse_transform(
        self, dataset: pd.DataFrame, settings: AppSettings
    ) -> pd.DataFrame:
        numeric_data_inverse_transformed = self._minmax_transformer.inverse_transform(
            dataset[settings.dataset.numeric_columns]
        )

        columns = self._onehot_transformer.transform_columns(
            settings.dataset.categorical_columns
        )
        categorical_data_inverse_transformed = (
            self._onehot_transformer.inverse_transform(dataset[columns])
        )

        ordinal_data_inverse_transformed = self._ordinal_transformer.inverse_transform(
            dataset[settings.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_inverse_transformed.join(
            categorical_data_inverse_transformed
        ).join(ordinal_data_inverse_transformed)

        dataset_transformed = dataset_transformed[settings.dataset.columns]

        return dataset_transformed

    @override
    def transform_column(self, column: str) -> list[str]:
        try:
            return self._onehot_transformer.transform_column(column)
        except Exception:
            return [column]


def download_dataset(settings: DatasetSettings) -> pd.DataFrame:
    _ = kagglehub.dataset_download(settings.kaggle_key, output_dir=settings.path)

    dataset = pd.read_csv(
        settings.file_path,
        header=None,
        delimiter=settings.delimiter,
        names=settings.columns,
    )

    return dataset
