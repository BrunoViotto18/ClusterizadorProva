from enum import Enum
import json
import os

from pydantic import BaseModel, PrivateAttr


class AppSettings(BaseModel):
    random_seed: int
    dataset: DatasetSettings
    normalizer: NormalizerSettings
    clusterizer: ClusterizerSettings

    @staticmethod
    def load_json(path: str) -> AppSettings:
        with open(path, "r") as f:
            json_settings = json.load(f)

        settings = AppSettings.model_validate(json_settings)

        return settings


class DatasetSettings(BaseModel):
    _path: list[str] = PrivateAttr(default=[])
    file_name: str
    delimiter: str
    columns: list[ColumnInfo]

    def __init__(self, **data):
        super().__init__(**data)

        self._path = data.pop("path", [])

    @property
    def path(self) -> str:
        return os.path.join(os.curdir, *self._path)

    @property
    def file_path(self) -> str:
        return os.path.join(self.path, self.file_name)

    @property
    def numeric_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.NUMERIC]

    @property
    def categorical_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.CATEGORICAL]

    @property
    def ordinal_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.ORDINAL]

    @property
    def column_names(self):
        return [x.name for x in self.columns]


class NormalizerSettings(BaseModel):
    _path: list[str] = PrivateAttr(default=[])
    file_name: str

    def __init__(self, **data):
        super().__init__(**data)

        self._path = data.pop("path", [])

    @property
    def path(self) -> str:
        return os.path.join(os.curdir, *self._path)

    @property
    def file_path(self) -> str:
        return os.path.join(self.path, self.file_name)


class ClusterizerSettings(BaseModel):
    _path: list[str] = PrivateAttr(default=[])
    file_name: str

    def __init__(self, **data):
        super().__init__(**data)

        self._path = data.pop("path", [])

    @property
    def path(self) -> str:
        return os.path.join(os.curdir, *self._path)

    @property
    def file_path(self) -> str:
        return os.path.join(self.path, self.file_name)


class ColumnInfo(BaseModel):
    name: str
    type: ColumnType
    fill: float | str | None = None


class ColumnType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class DatasetTransformerMetadata(BaseModel):
    dataset: DatasetMetadata

    @staticmethod
    def load_json(path: str) -> DatasetTransformerMetadata:
        with open(path, "r") as f:
            json_settings = json.load(f)

        settings = DatasetTransformerMetadata.model_validate(json_settings)

        return settings


class DatasetMetadata(BaseModel):
    columns: list[ColumnInfo]
    transformed_columns: list[str]

    @property
    def numeric_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.NUMERIC]

    @property
    def categorical_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.CATEGORICAL]

    @property
    def ordinal_columns(self):
        return [x.name for x in self.columns if x.type == ColumnType.ORDINAL]

    @property
    def column_names(self):
        return [x.name for x in self.columns]
