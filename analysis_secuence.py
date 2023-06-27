import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from io import BytesIO
import base64
from fastapi.responses import StreamingResponse,Response
from sklearn.metrics import silhouette_samples, silhouette_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay

buffer = BytesIO()

sns.set(
    rc={
        "figure.figsize": (10, 10)
    }
)

sns.set_style("whitegrid")

dataset = pd.read_csv("data/wine_quality.csv")
dataset_numcols = dataset.select_dtypes(include=[float, int])
correlation_matrix = dataset_numcols.corr()
dataset_numcols.drop(["good", "quality"], axis=1, inplace=True)
km_dw_numcols = KMeans(n_clusters=2, n_init=20).fit(dataset_numcols)
scaled_dataset = pd.DataFrame(StandardScaler().fit_transform(dataset_numcols), columns=dataset_numcols.columns)
scaled_dataset_for_predict = pd.concat([scaled_dataset, dataset["color"]], axis=1)
random_dataset = dataset_numcols.apply(lambda x: np.random.uniform(min(x), max(x), len(x)))
scaled_random_dataset = pd.DataFrame(scale(random_dataset))
distance_matrix_dw = squareform(pdist(dataset_numcols)) ** 2
distance_matrix_rdw = squareform(pdist(random_dataset)) ** 2


def graph_missingness_matrix():
    msno.matrix(dataset)


def generate_all_histograms():
    for c in dataset.columns:
        if c != "good" and c != "color":
            colHist = sns.histplot(data=dataset, x=c, hue="color", bins=75, kde=False,
                                   palette={"red": "red", "white": "white"})
            colHist.set(title="Count of \"" + str(c) + "\" by color")
            plt.show()


def generate_all_barplots():
    for c in dataset.columns:
        if c != "quality" and c != "good" and c != "color":
            colHist = sns.barplot(data=dataset, x="quality", y=c, hue="color", palette={"red": "red", "white": "white"},
                                  edgecolor="black")
            colHist.set(title=f"Distribution of \"{str(c)}\" by color")
            plt.show()


def graph_correlation_matrix():
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f", center=0, cbar_kws={"shrink": .5})
    plt.title("Correlation Matrix")
    plt.show()


def pca_graph():
    pca = PCA(n_components=2)
    reduced_pca_components = pca.fit_transform(scaled_dataset)
    plt.scatter(reduced_pca_components[:, 0], reduced_pca_components[:, 1], c=km_dw_numcols.labels_, cmap='viridis')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Análisis de Clústeres')
    plt.savefig(buffer, format='svg')
    buffer.seek(0)
    response = Response(content=buffer.getvalue(), media_type='image/jpeg')
    response.headers['Content-Disposition'] = 'attachment; filename=plot.jpg'
    return response


def scatter_graph():
    scaled_dw_plot = PCA(n_components=2).fit_transform(scaled_dataset)
    plt.scatter(scaled_dw_plot[:, 0], scaled_dw_plot[:, 1], c=dataset["color"])
    plt.title("PCA - Wine Quality Data")
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def scatter_random_graph():
    scaled_rdw_plot = PCA(n_components=2).fit_transform(scaled_random_dataset)
    plt.scatter(scaled_rdw_plot[:, 0], scaled_rdw_plot[:, 1], c=dataset["color"])
    plt.title("PCA - Random Data")
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def hierarchical_graph():
    hclust_dw = linkage(distance_matrix_dw, method='ward')

    hclust_rdw = linkage(distance_matrix_rdw, method='ward')

    plt.subplot(2, 1, 1)
    dend_dw = dendrogram(hclust_dw, no_labels=True)
    plt.title("Hierarchical Clustering - Wine Quality Data")

    plt.subplot(2, 1, 2)
    dend_rdw = dendrogram(hclust_rdw, no_labels=True)
    plt.title("Hierarchical Clustering - Random Data")

    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def hopkins(data):
    nbrs = NearestNeighbors(n_neighbors=1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    random_data = np.random.uniform(low=min(data), high=max(data), size=data.shape)
    random_nbrs = NearestNeighbors(n_neighbors=1).fit(random_data)
    random_distances, _ = random_nbrs.kneighbors(random_data)

    hopkins_statistic = np.sum(distances) / (np.sum(distances) + np.sum(random_distances))
    return hopkins_statistic


def perform_hopkin_statistic():
    hop_stat_dw = hopkins(scaled_dataset)
    print("Hopkins Statistic - Wine Quality Data:", hop_stat_dw)

    hop_stat_rdw = hopkins(scaled_random_dataset)
    print("Hopkins Statistic - Random Data:", hop_stat_rdw)


def graph_elbow():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init="auto")
        kmeans.fit(dataset_numcols)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')


def graph_silhouette():
    silhouette_dw = silhouette_samples(distance_matrix_dw, km_dw_numcols.labels_)
    silhouette_avg_dw = silhouette_score(distance_matrix_dw, km_dw_numcols.labels_)
    print("Average Silhouette Score - Wine Quality Data:", silhouette_avg_dw)
    n_clusters = len(np.unique(km_dw_numcols.labels_))
    y_lower = 10
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        cluster_silhouette_values = silhouette_dw[km_dw_numcols.labels_ == i]
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
    plt.title("Silhouette Analysis - Wine Quality Data")
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return StreamingResponse(BytesIO(base64.b64decode(base64_image)), media_type='image/png')
