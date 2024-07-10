import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

NUM_POINTS = 500


def save_plot(plt, question, name):
    output_dir = os.path.join('../output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"Question2{question}_{name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# Load datasets generated in Problem 1
def load_dataset_A():
    x = np.random.uniform(-10, 2, NUM_POINTS)
    y = np.random.uniform(18, 45, NUM_POINTS)
    return np.column_stack((x, y))


def load_dataset_B():
    centers = [[1, -1], [2, -2], [4, -4]]
    stds = [0.5 * i for i in [1, 2, 4]]
    x, y = [], []
    for (cx, cy), std in zip(centers, stds):
        x.extend(np.random.normal(cx, std, NUM_POINTS // 3))
        y.extend(np.random.normal(cy, std, NUM_POINTS // 3))
    return np.column_stack((x, y))


def generate_y(offset_x=0, offset_y=0):
    points = []
    for i in range(20):
        points.append((offset_x + 0, offset_y + i))
    for i in range(20):
        points.append((offset_x - i, offset_y + 20 + i))
    for i in range(20):
        points.append((offset_x + i, offset_y + 20 + i))
    points = np.array(points) + np.random.normal(0, 0.5, (len(points), 2))
    return points


def generate_s(offset_x=0, offset_y=0):
    points = []
    for i in range(20):
        points.append((offset_x + i, offset_y + 20))
    for i in range(20):
        points.append((offset_x + i, offset_y + 20 - i))
    for i in range(20):
        points.append((offset_x + i, offset_y + 0))
    points = np.array(points) + np.random.normal(0, 0.5, (len(points), 2))
    return points


def load_dataset_C():
    y1_points = generate_y(-30, 0)
    y2_points = generate_y(0, 0)
    y3_points = generate_y(30, 0)
    s_points = generate_s(60, 0)
    return np.concatenate((y1_points, y2_points, y3_points, s_points), axis=0)


def load_dataset_D():
    x_positions = [0, 5, 0, 5]
    y_positions = [0, 0, 2, 2]
    x, y = [], []
    for i in range(4):
        x.extend(np.random.normal(x_positions[i], 0.5, NUM_POINTS // 4))
        y.extend(np.random.normal(y_positions[i], 0.1, NUM_POINTS // 4))
    return np.column_stack((x, y))


def load_dataset_E():
    x, y = make_moons(n_samples=NUM_POINTS, noise=0.05)
    x[:, 0] *= 1.2
    return x


def load_dataset_F():
    x, y = make_moons(n_samples=NUM_POINTS, noise=0.1)
    idx = np.random.choice(len(x), size=50, replace=False)
    x[idx] += np.random.normal(0, 1, (50, 2))
    return x


datasets = {
    'A': load_dataset_A(),
    'B': load_dataset_B(),
    'C': load_dataset_C(),
    'D': load_dataset_D(),
    'E': load_dataset_E(),
    'F': load_dataset_F(),
}


def plot_clusters(data, labels, title, question, name):
    plt.figure(figsize=(10, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    save_plot(plt, question, name)


### QUESTION A ###
def kmeans_clustering(dataset_key, question):
    data = datasets[dataset_key]
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data)
        title = f'K-means Clustering (k={k})'
        name = f'KMEANS_k{k}'
        plot_clusters(data, labels, title, question, name)


### QUESTION B ###
def hierarchical_clustering(dataset_key, question):
    data = datasets[dataset_key]
    linkages = ['single', 'complete', 'average']
    for linkage in linkages:
        for k in [2, 4]:
            hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = hc.fit_predict(data)
            title = f'Hierarchical Clustering ({linkage} linkage, k={k})'
            name = f'HC_{linkage}_k{k}'
            plot_clusters(data, labels, title, question, name)


for dataset_key in datasets.keys():
    kmeans_clustering(dataset_key, dataset_key)
    hierarchical_clustering(dataset_key, dataset_key)
