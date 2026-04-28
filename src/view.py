import argparse
import pickle

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

    normalized_centers = pd.DataFrame(cluster_model.cluster_centers_, columns=normalizer.normalized_columns)

    print("Centros normalizados")
    print(normalized_centers)

    natural_centers = normalizer.denormalize(normalized_centers)

    print("Centros naturais")
    print(natural_centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clusterizer on dataset")

    parser.add_argument("settings_path", help="Path to json settings")

    args = parser.parse_args()

    main(args.settings_path)
