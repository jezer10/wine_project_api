import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyclustertend import hopkins
from sklearn.metrics import silhouette_samples, silhouette_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sns.set(
    rc={
        "figure.figsize": (10, 10)
    }
)

sns.set_style("whitegrid")

dataset = pd.read_csv("./datasets/winequality.csv")
dataset.head()

dataset.shape

dataset.info()

dataset.describe()

dataset.isna().sum()

dataset.empty

msno.matrix(dataset)

for c in dataset.columns:
    if c != "good" and c != "color":
        colHist = sns.histplot(data=dataset, x=c, hue="color", bins=75, kde=False,
                               palette={"red": "red", "white": "white"})
        colHist.set(title="Count of \"" + str(c) + "\" by color")
        plt.show()

for c in dataset.columns:
    if c != "quality" and c != "good" and c != "color":
        colHist = sns.barplot(data=dataset, x="quality", y=c, hue="color", palette={"red": "red", "white": "white"},
                              edgecolor="black")
        colHist.set(title=f"Distribution of \"{str(c)}\" by color")
        plt.show()

dataset_numcols = dataset.select_dtypes(include=[float, int])
dataset_numcols

correlation_matrix = dataset_numcols.corr()
correlation_matrix

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, annot=True, fmt=".2f", center=0, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix")
plt.show()

dataset_numcols.drop(["good", "quality"], axis=1, inplace=True)
scaler = StandardScaler()
scaled_dataset = pd.DataFrame(scaler.fit_transform(dataset_numcols), columns=dataset_numcols.columns)
print(scaled_dataset.shape)
print(scaled_dataset.head())

scaled_dataset_for_predict = pd.concat([scaled_dataset, dataset["color"]], axis=1)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init="auto")
    kmeans.fit(dataset_numcols)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

km_dw_numcols = KMeans(n_clusters=2, n_init=20).fit(dataset_numcols)
print(km_dw_numcols)

pd.crosstab(km_dw_numcols.labels_, dataset['color'])

pca = PCA(n_components=2)
reduced_pca_componets = pca.fit_transform(scaled_dataset)

plt.scatter(reduced_pca_componets[:, 0], reduced_pca_componets[:, 1], c=km_dw_numcols.labels_, cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Análisis de Clústeres')
plt.show()

random_dataset = dataset_numcols.apply(lambda x: np.random.uniform(min(x), max(x), len(x)))
random_dataset.head()

scaled_random_dataset = pd.DataFrame(scale(random_dataset))
print(scaled_random_dataset.shape)
scaled_random_dataset.head()

scaled_dw_plot = PCA(n_components=2).fit_transform(scaled_dataset)
scaled_rdw_plot = PCA(n_components=2).fit_transform(scaled_random_dataset)

plt.subplot(2, 1, 1)
plt.scatter(scaled_dw_plot[:, 0], scaled_dw_plot[:, 1], c=dataset["color"])
plt.title("PCA - Wine Quality Data")
plt.subplot(2, 1, 2)
plt.scatter(scaled_rdw_plot[:, 0], scaled_rdw_plot[:, 1], c=dataset["color"])
plt.title("PCA - Random Data")
plt.tight_layout()
plt.show()

distance_matrix_dw = squareform(pdist(dataset_numcols)) ** 2
distance_matrix_rdw = squareform(pdist(random_dataset)) ** 2

# Realizar el agrupamiento jerárquico para el conjunto de datos original
hclust_dw = linkage(distance_matrix_dw, method='ward')

# Realizar el agrupamiento jerárquico para el conjunto de datos aleatorio
hclust_rdw = linkage(distance_matrix_rdw, method='ward')

# Graficar los dendrogramas
plt.subplot(2, 1, 1)
dend_dw = dendrogram(hclust_dw, no_labels=True)
plt.title("Hierarchical Clustering - Wine Quality Data")

plt.subplot(2, 1, 2)
dend_rdw = dendrogram(hclust_rdw, no_labels=True)
plt.title("Hierarchical Clustering - Random Data")

plt.tight_layout()
plt.show()

# Calcular la estadística de Hopkins para el conjunto de datos original
hop_stat_dw = hopkins(scaled_dataset, sampling_size=scaled_dataset.shape[0] - 1)
print("Hopkins Statistic - Wine Quality Data:", hop_stat_dw)

# Calcular la estadística de Hopkins para el conjunto de datos aleatorio
hop_stat_rdw = hopkins(scaled_random_dataset, sampling_size=scaled_random_dataset.shape[0] - 1)
print("Hopkins Statistic - Random Data:", hop_stat_rdw)

silhouette_dw = silhouette_samples(distance_matrix_dw, km_dw_numcols.labels_)
silhouette_avg_dw = silhouette_score(distance_matrix_dw, km_dw_numcols.labels_)
print("Average Silhouette Score - Wine Quality Data:", silhouette_avg_dw)

# Graficar el coeficiente de silueta
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
plt.show()

X = scaled_dataset_for_predict.drop("color", axis=1)  # Variables predictoras
y = scaled_dataset_for_predict["color"]  # Variable objetivo

# Realizar la división
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Crear el modelo de SVM
svm_model = SVC(kernel='rbf')

# Ajustar el modelo con los datos de entrenamiento
svm_model.fit(X_train, y_train)

# Imprimir un resumen del modelo
print(svm_model)

# Realizar predicciones en el conjunto de prueba
svm_predictions = svm_model.predict(X_test)

# Imprimir el informe de clasificación
print(classification_report(y_test, svm_predictions))

# Calcular la matriz de confusión
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix, display_labels=svm_model.classes_)
disp.plot()

# Crear el modelo de Random Forest
randomforest_model = RandomForestClassifier()

# Ajustar el modelo con los datos de entrenamiento
randomforest_model.fit(X_train, y_train)

# Visualizar la importancia de las variables
importance = randomforest_model.feature_importances_
print(importance)

# Realizar predicciones en el conjunto de prueba
randomforest_predictions = randomforest_model.predict(X_test)

# Imprimir el informe de clasificación
print(classification_report(y_test, randomforest_predictions))

# Calcular la matriz de confusión
randomforest_confusion_matrix = confusion_matrix(y_test, randomforest_predictions)

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=randomforest_confusion_matrix,
                              display_labels=randomforest_model.classes_)
disp.plot()

scaled_dataset_for_predict.drop("color", axis=1)

scaled_dataset_for_predict.iloc[0:1, ]