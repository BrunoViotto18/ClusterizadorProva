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
    kaggle_key: str
    _path: list[str] = PrivateAttr(default=[])
    file_name: str
    delimiter: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    ordinal_columns: list[str]
    columns: list[str]

    def __init__(self, **data):
        super().__init__(**data)

        self._path = data.pop("path", [])

    @property
    def path(self) -> str:
        return os.path.join(os.curdir, *self._path)

    @property
    def file_path(self) -> str:
        return os.path.join(self.path, self.file_name)


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
