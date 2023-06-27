from sklearn.preprocessing import StandardScaler
import pandas as pd
from models import decision_tree, random_forest, support_vector_machine





def scale_wines_dataset(data):
    numeric_columns = pd.DataFrame(data.dict()["wines"]).select_dtypes(include=[float, int])
    scaled_wines_dataset = pd.DataFrame(StandardScaler().fit_transform(numeric_columns),
                                        columns=numeric_columns.columns)
    return scaled_wines_dataset


def predict_by_tree_model(data):
    scaled_wines_dataset = scale_wines_dataset(data)
    predictions = decision_tree.model.predict(scaled_wines_dataset)
    scaled_wines_dataset["color"] = predictions
    return scaled_wines_dataset.to_dict(orient="records")


def predict_by_svm(data):
    scaled_wines_dataset = scale_wines_dataset(data)
    predictions = support_vector_machine.model.predict(scaled_wines_dataset)
    scaled_wines_dataset["color"] = predictions
    return scaled_wines_dataset.to_dict(orient="records")


def predict_by_random_forest(data):
    scaled_wines_dataset = scale_wines_dataset(data)
    predictions = random_forest.model.predict(scaled_wines_dataset)
    scaled_wines_dataset["color"] = predictions
    return scaled_wines_dataset.to_dict(orient="records")
