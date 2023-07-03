import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np


@pd.api.extensions.register_dataframe_accessor("utilities")
class UtilitiesAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def capitalize(self):
        return self._obj.rename(
            columns=lambda x: x.split()[0] + ''.join(word.capitalize() for word in x.split()[1:])
        )

    def numeric(self):
        return self._obj.select_dtypes(include=[float, int])


dataset = pd.read_csv("data/wine_quality.csv").utilities.capitalize()
dataset_numeric_columns = dataset.utilities.numeric()

dataset_numeric_columns.drop(["good", "quality"], axis=1, inplace=True)

scaled_dataset = pd.DataFrame(StandardScaler().fit_transform(dataset_numeric_columns),
                              columns=dataset_numeric_columns.columns)
dataset_for_predict = pd.concat([scaled_dataset, dataset["color"]], axis=1)

X = dataset_for_predict.drop("color", axis=1)

y = pd.concat([dataset[["quality", "good"]], dataset["color"].replace({"red": 0, "white": 1})], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


def get_dataset_mean():
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(scaled_dataset)
    principal_components = pca.components_
    results = {
        'white': [],
        'red': []
    }
    for j, component in enumerate(scaled_dataset.columns):
        component_values = []
        for i in range(n_components):
            component_values.append(principal_components[i][j])
        results[list(results.keys())[np.argmax(component_values)]].append({
            'name': component,
            'max': dataset[component].max().round(2),
            'min': dataset[component].min().round(2),
            'avg': dataset[component].mean().round(2),
        })

    return results


def get_numeric_dataset(color: str = None):
    results = {}
    filtered_dataset = dataset[dataset["color"] == color] if color in ("white", "red") else dataset
    ds_cl = filtered_dataset.utilities.numeric()
    ds_cl.drop(["good", "quality"], axis=1, inplace=True)
    for column in ds_cl.columns:
        results[column] = {
            'max': ds_cl[column].max().round(2),
            'min': ds_cl[column].min().round(2),
            'average': ds_cl[column].mean().round(2)
        }
    return results


