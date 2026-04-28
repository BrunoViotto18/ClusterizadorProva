from typing import Any, Protocol

import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class SimpleTransformer(Protocol):
    def fit(self, X: Any, y: Any = None) -> Any: ...
    def transform(self, X: Any) -> Any: ...
    def inverse_transform(self, X: Any) -> Any: ...


class DatasetTransformer:

    def __init__(self, scaler: SimpleTransformer):
        self._scaler: SimpleTransformer = scaler

    @staticmethod
    def fit_save(
        file_path: str, dataset: pd.DataFrame, transformer: SimpleTransformer
    ) -> DatasetTransformer:
        transformer = transformer.fit(dataset)

        with open(file_path, "wb") as f:
            pickle.dump(transformer, f)

        return DatasetTransformer(transformer)

    @staticmethod
    def load(file_path: str) -> DatasetTransformer:
        with open(file_path, "rb") as f:
            scaler = pickle.load(f)

        return DatasetTransformer(scaler)

    def get_normalized_columns(self, columns: np.ndarray) -> np.ndarray:
        if isinstance(self._scaler, OneHotEncoder):
            columns = self._scaler.get_feature_names_out(columns)

        return columns

    def transform(self, dataset: pd.DataFrame):
        columns = dataset.columns

        transformed_data = self._scaler.transform(dataset)

        if isinstance(self._scaler, OneHotEncoder):
            columns = self._scaler.get_feature_names_out(columns)
            transformed_data = transformed_data.toarray()

        transformed_dataset = pd.DataFrame(transformed_data, columns=columns)

        return transformed_dataset

    def inverse_transform(
        self, dataset: pd.DataFrame, columns: np.ndarray | None = None
    ):
        if isinstance(self._scaler, OneHotEncoder):
            columns = self._scaler.get_feature_names_out(columns)
            dataset = dataset[columns]
        else:
            columns = np.array(dataset.columns)

        reverse_transformed_data = self._scaler.inverse_transform(dataset)

        if isinstance(self._scaler, OneHotEncoder):
            columns = self._scaler.feature_names_in_

        reverse_transformed_dataset = pd.DataFrame(
            reverse_transformed_data, columns=columns
        )

        return reverse_transformed_dataset
