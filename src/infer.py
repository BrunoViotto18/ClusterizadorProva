import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from dataset import DatasetNormalizer
from settings import AppSettings, ClusterizerSettings


def load_cluster_model(settings: ClusterizerSettings) -> KMeans:
    with open(settings.file_path, "rb") as f:
        model: KMeans = pickle.load(f)

    return model


def main(settings_path: str) -> None:
    settings = AppSettings.load_json(settings_path)

    cluster_model = load_cluster_model(settings.clusterizer)

    normalizer = DatasetNormalizer.load(settings)

    new_data = pd.DataFrame(
        [
            [
                0.00632,
                18.00,
                2.310,
                0,
                0.5380,
                6.5750,
                65.20,
                4.0900,
                1,
                296.0,
                15.30,
                396.90,
                4.98,
                24.00,
            ]
        ],
        columns=settings.dataset.columns,
    )

    print(new_data)

    normalized_data = normalizer.transform(new_data)

    print(normalized_data)

    cluster_index = cluster_model.predict(normalized_data)

    print(f"Predicted cluster {cluster_index}")

    predicted = pd.DataFrame(
        cluster_model.cluster_centers_[cluster_index],
        columns=normalizer.normalized_columns,
    )

    print(predicted)

    natural_prediction = normalizer.inverse_transform(predicted)

    print(natural_prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clusterizer on dataset")

    parser.add_argument("settings_path", help="Path to json settings")

    args = parser.parse_args()

    main(args.settings_path)
