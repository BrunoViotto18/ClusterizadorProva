import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from transformer import DatasetTransformer
from settings import AppSettings, ClusterizerSettings


def load_cluster_model(settings: ClusterizerSettings) -> KMeans:
    with open(settings.file_path, "rb") as f:
        model: KMeans = pickle.load(f)

    return model


def main(settings_path: str) -> None:
    settings = AppSettings.load_json(settings_path)

    cluster_model = load_cluster_model(settings.clusterizer)

    normalizer = DatasetTransformer.load(settings.normalizer.file_path)

    new_data = pd.DataFrame(
        [
            [
                "Female",
                21,
                1.62,
                64,
                "yes",
                "no",
                2,
                3,
                "Sometimes",
                "no",
                2,
                "no",
                0,
                1,
                "no",
                "Public_Transportation",
                "Normal_Weight",
            ]
        ],
        columns=normalizer.original_columns,
    )

    print("Novo dado")
    print(new_data)

    normalized_data = normalizer.transform(new_data)

    print("Novo dado normalizado")
    print(normalized_data)

    cluster_index = cluster_model.predict(normalized_data)

    print(f"Cluster predito {cluster_index}")

    predicted = pd.DataFrame(
        cluster_model.cluster_centers_[cluster_index],
        columns=normalizer.transformed_columns,
    )

    print(f"Valores do cluster {cluster_index}")
    print(predicted)

    natural_prediction = normalizer.inverse_transform(predicted)

    print(f"Valores do cluster naturalizados {cluster_index}")
    print(natural_prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clusterizer on dataset")

    parser.add_argument("settings_path", help="Path to json settings")

    args = parser.parse_args()

    main(args.settings_path)
