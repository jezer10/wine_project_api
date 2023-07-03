from models.load_dataset import dataset, dataset_numeric_columns, scaled_dataset
import pandas as pd
import missingno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from io import BytesIO
import base64
from fastapi.responses import StreamingResponse, Response
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.datasets import load_wine

sns.set(
    rc={
        "figure.figsize": (10, 10)
    }
)

sns.set_style("whitegrid")

correlation_matrix = dataset_numeric_columns.corr()
km_dw_numeric_columns = KMeans(n_clusters=2, n_init=20).fit(dataset_numeric_columns)
random_dataset = dataset_numeric_columns.apply(lambda x: np.random.uniform(min(x), max(x), len(x)))
scaled_random_dataset = pd.DataFrame(scale(random_dataset))
distance_matrix_dw = squareform(pdist(dataset_numeric_columns)) ** 2
distance_matrix_rdw = squareform(pdist(random_dataset)) ** 2


def missing_matrix_graph():
    missingno.matrix(dataset)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


def histogram_graph(column: str):
    if column != "good" and column != "color":
        fig, ax = plt.subplots()
        column_histogram = sns.histplot(data=dataset, x=column, hue="color", bins=75, kde=False,
                                        palette={"red": "red", "white": "white"}, ax=ax)
        column_histogram.set(title="Count of \"" + str(column) + "\" by color")
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")


def barplot_graph(column: str):
    if column != "quality" and column != "good" and column != "color":
        fig, ax = plt.subplots()
        column_histogram = sns.barplot(data=dataset, x="quality", y=column, hue="color",
                                       palette={"red": "red", "white": "white"}, edgecolor="black", ax=ax)
        column_histogram.set(title=f"Distribution of \"{str(column)}\" by color")
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")


def graph_correlation_matrix():
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f", center=0, cbar_kws={"shrink": .5})
    plt.title("Correlation Matrix")


def pca_graph():
    pca = PCA(n_components=2)
    reduced_pca_components = pca.fit_transform(scaled_dataset)
    print(reduced_pca_components.shape)
    fig, ax = plt.subplots()
    ax.scatter(reduced_pca_components[:, 0],
               reduced_pca_components[:, 1],
               c=km_dw_numeric_columns.labels_,
               cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Cluster Analysis')
    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def graph_elbow():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init="auto")
        kmeans.fit(dataset_numeric_columns)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('The Elbow Method')
    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def scatter_graph():
    scaled_dw_plot = PCA(n_components=2).fit_transform(scaled_dataset)
    fig, ax = plt.subplots()
    ax.scatter(scaled_dw_plot[:, 0], scaled_dw_plot[:, 1], c=dataset["color"])
    ax.set_title("PCA - Wine Quality Data")
    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def scatter_random_graph():
    scaled_rdw_plot = PCA(n_components=2).fit_transform(scaled_random_dataset)
    fig, ax = plt.subplot()
    ax.scatter(scaled_rdw_plot[:, 0], scaled_rdw_plot[:, 1], c=dataset["color"])
    ax.title("PCA - Random Data")
    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def hierarchical_graph():
    # hierarchical_cluster_dw = linkage(distance_matrix_dw, method="ward")
    # dendrogram(hierarchical_cluster_dw, no_labels=True)
    # plt.title("Hierarchical Clustering - Wine Quality Data")
    # FigureCanvas(plt).print_png(buffer)
    # base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # plt.close(plt)
    # return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')
    return True


def hierarchical_random_graph():
    # hierarchical_cluster_rdw = linkage(distance_matrix_rdw, method='ward')
    # dendrogram(hierarchical_cluster_rdw, no_labels=True)
    # plt.title("Hierarchical Clustering - Random Data")
    # FigureCanvas(plt).print_png(buffer)
    # base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # plt.close(plt)
    # return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')
    return True


def hopkins(data):
    neighbors = NearestNeighbors(n_neighbors=1).fit(data)
    distances, _ = neighbors.kneighbors(data)
    random_data = np.random.uniform(low=min(data), high=max(data), size=data.shape)
    random_neighbors = NearestNeighbors(n_neighbors=1).fit(random_data)
    random_distances, _ = random_neighbors.kneighbors(random_data)
    hopkins_statistic = np.sum(distances) / (np.sum(distances) + np.sum(random_distances))
    return hopkins_statistic


def perform_hopkins_statistic():
    hop_stat_dw = hopkins(scaled_dataset)
    print("Hopkins Statistic - Wine Quality Data:", hop_stat_dw)
    hop_stat_rdw = hopkins(scaled_random_dataset)
    print("Hopkins Statistic - Random Data:", hop_stat_rdw)


def silhouette_graph():
    silhouette_dw = silhouette_samples(distance_matrix_dw, km_dw_numeric_columns.labels_)
    silhouette_avg_dw = silhouette_score(distance_matrix_dw, km_dw_numeric_columns.labels_)
    print("Average Silhouette Score - Wine Quality Data:", silhouette_avg_dw)
    n_clusters = len(np.unique(km_dw_numeric_columns.labels_))
    y_lower = 10
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        cluster_silhouette_values = silhouette_dw[km_dw_numeric_columns.labels_ == i]
        cluster_silhouette_values.sort()
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.get_cmap("tab10")(i)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg_dw, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_title("Silhouette Analysis - Wine Quality Data")
    buffer = BytesIO()
    FigureCanvas(fig).print_png(buffer)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def get_dataset_columns():
    list_columns = dataset.columns.tolist()
    return {
        "input_variables": list_columns[:-3],
        "output_variables": list_columns[-3:]
    }
