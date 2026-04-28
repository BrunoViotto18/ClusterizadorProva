import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import DatasetTransformer, download_dataset
from scipy.spatial.distance import cdist
from settings import AppSettings, DatasetSettings
from sklearn.cluster import KMeans


def calculate_ideal_cluster_count(dataset: pd.DataFrame, random_seed: int):
    distortions = []
    K = range(1, dataset.shape[0])
    for k in K:
        cluster_model = KMeans(n_clusters=k, random_state=random_seed)

        cluster_model = cluster_model.fit(dataset)

        dist = cdist(dataset, cluster_model.cluster_centers_, "euclidean")
        distortion = sum(np.min(dist, axis=1) / dataset.shape[0])
        distortions.append(distortion)

    _, ax = plt.subplots()

    ax.plot(K, distortions)
    ax.set(
        xlabel="n Clusters", ylabel="Distortions", title="Elbow Method for Optimal k"
    )

    plt.show()

    x0 = K[0]
    y0 = distortions[0]
    xn = K[-1]
    yn = distortions[-1]
    distances = []
    for i in range(len(distortions)):
        x = K[i]
        y = distortions[i]
        numerator = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
        denominator = np.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
        distance = numerator / denominator
        distances.append(distance)

    longest_distance = np.max(distances)

    longest_distance_index = distances.index(longest_distance)

    optimized_cluster_count = K[longest_distance_index]

    return optimized_cluster_count


def train_cluster_model(
    dataset: pd.DataFrame, cluster_count: int, random_seed: int
) -> KMeans:
    cluster_model = KMeans(n_clusters=cluster_count, random_state=random_seed)

    cluster_model.fit(dataset)

    return cluster_model


def save_cluster_model(file_path: str, cluster_model: KMeans):
    parent_directory = os.path.dirname(file_path)
    os.makedirs(parent_directory, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(cluster_model, f)


def main(settings_path: str) -> None:
    settings = AppSettings.load_json(settings_path)

    dataset = download_dataset(settings.dataset)

    dataset_normalizer = DatasetTransformer.fit_save(
        settings.normalizer.file_path, dataset, settings
    )

    dataset_transformed = dataset_normalizer.transform(dataset, settings)

    ideal_cluster_count = calculate_ideal_cluster_count(
        dataset_transformed, settings.random_seed
    )

    print(ideal_cluster_count)

    cluster_model = train_cluster_model(
        dataset_transformed, ideal_cluster_count, settings.random_seed
    )

    save_cluster_model(settings.clusterizer.file_path, cluster_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train clusterizer on dataset")

    parser.add_argument("settings_path", help="Path to json settings")

    args = parser.parse_args()

    main(args.settings_path)
