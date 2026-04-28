import json
import os
import numpy as np
import pandas as pd
import pickle

from abc import ABC, abstractmethod

from settings import (
    ColumnType,
    DatasetMetadata,
    DatasetTransformerMetadata,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from typing import Any, override


class Transfomer(ABC):

    @staticmethod
    @abstractmethod
    def fit_save(
        path: str, dataset: pd.DataFrame, meta: DatasetMetadata | None = None
    ) -> Transfomer:
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
        path: str, dataset: pd.DataFrame, meta: DatasetMetadata | None = None
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
        path: str, dataset: pd.DataFrame, meta: DatasetMetadata | None = None
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
        return [*self._scaler.get_feature_names_out()]

    def transform_columns(self, columns: list[str]) -> list[str]:
        return [*self._scaler.get_feature_names_out()]


class OrdinalTransformer(Transfomer):
    def __init__(self, scaler: OrdinalEncoder):
        self._scaler: OrdinalEncoder = scaler

    @staticmethod
    @override
    def fit_save(
        path: str, dataset: pd.DataFrame, meta: DatasetMetadata | None = None
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
        meta: DatasetTransformerMetadata,
        minmax_transformer: MinMaxTransformer,
        onehot_transformer: OneHotTransformer,
        ordinal_transformer: OrdinalTransformer,
    ):
        self._meta: DatasetTransformerMetadata = meta
        self._minmax_transformer: MinMaxTransformer = minmax_transformer
        self._onehot_transformer: OneHotTransformer = onehot_transformer
        self._ordinal_transformer: OrdinalTransformer = ordinal_transformer

    @staticmethod
    def fit_save(
        path: str, dataset: pd.DataFrame, meta: DatasetMetadata | None = None
    ) -> DatasetTransformer:

        if meta is None:
            raise ValueError("No dataset metada found")

        for column in meta.columns:
            fill: float
            if column.type == ColumnType.NUMERIC:
                dataset[column.name] = pd.to_numeric(
                    dataset[column.name], errors="coerce"
                )
                fill = dataset[column.name].median()
            elif column.type == ColumnType.CATEGORICAL:
                fill = dataset[column.name].mode()[0]
            elif column.type == ColumnType.ORDINAL:
                fill = dataset[column.name].mode()[0]
            else:
                raise ValueError("Unknown column type")

            dataset[column.name] = dataset[column.name].fillna(fill)
            column.fill = fill

        parent_directory = os.path.dirname(path)
        os.makedirs(parent_directory, exist_ok=True)

        minmax_transformer = MinMaxTransformer.fit_save(
            os.path.join(parent_directory, "minmax_transformer.pkl"),
            dataset[meta.numeric_columns],
            meta,
        )

        onehot_transformer = OneHotTransformer.fit_save(
            os.path.join(parent_directory, "onehot_transformer.pkl"),
            dataset[meta.categorical_columns],
            meta,
        )

        ordinal_transformer = OrdinalTransformer.fit_save(
            os.path.join(parent_directory, "ordinal_transformer.pkl"),
            dataset[meta.ordinal_columns],
            meta,
        )

        meta.transformed_columns = [
            *meta.numeric_columns,
            *onehot_transformer.transform_columns(meta.categorical_columns),
            *meta.ordinal_columns,
        ]
        normalizer_meta = DatasetTransformerMetadata(dataset=meta)
        with open(path, "w") as f:
            json.dump(normalizer_meta.model_dump(mode="json"), f)

        return DatasetTransformer(
            normalizer_meta,
            minmax_transformer,
            onehot_transformer,
            ordinal_transformer,
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

        meta = DatasetTransformerMetadata.load_json(path)

        return DatasetTransformer(
            meta,
            minmax_transformer,
            onehot_transformer,
            ordinal_transformer,
        )

    @property
    def original_columns(self):
        return self._meta.dataset.column_names

    @property
    def transformed_columns(self):
        return self._meta.dataset.transformed_columns

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        for column in self._meta.dataset.columns:
            dataset[column.name] = dataset[column.name].fillna(column.fill)

        numeric_data_transformed = self._minmax_transformer.transform(
            dataset[self._meta.dataset.numeric_columns]
        )

        categorical_data_transformed = self._onehot_transformer.transform(
            dataset[self._meta.dataset.categorical_columns]
        )

        ordinal_data_transformed = self._ordinal_transformer.transform(
            dataset[self._meta.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_transformed.join(
            categorical_data_transformed
        ).join(ordinal_data_transformed)

        return dataset_transformed

    def inverse_transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        numeric_data_inverse_transformed = self._minmax_transformer.inverse_transform(
            dataset[self._meta.dataset.numeric_columns]
        )

        columns = self._onehot_transformer.transform_columns(
            self._meta.dataset.categorical_columns
        )
        categorical_data_inverse_transformed = (
            self._onehot_transformer.inverse_transform(dataset[columns])
        )

        ordinal_data_inverse_transformed = self._ordinal_transformer.inverse_transform(
            dataset[self._meta.dataset.ordinal_columns]
        )

        dataset_transformed = numeric_data_inverse_transformed.join(
            categorical_data_inverse_transformed
        ).join(ordinal_data_inverse_transformed)

        dataset_transformed = dataset_transformed[self._meta.dataset.column_names]

        return dataset_transformed

    @override
    def transform_column(self, column: str) -> list[str]:
        # try:
        return self._onehot_transformer.transform_column(column)

    # except Exception:
    #     return [column]
