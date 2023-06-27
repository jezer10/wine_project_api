import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

y = dataset_for_predict["color"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
