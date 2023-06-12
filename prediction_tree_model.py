import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def parseDataFrameToCamelCase(df):
    return df.rename(
        columns=lambda x: x.split()[0] + ''.join(word.capitalize() for word in x.split()[1:])
    )


dataset = parseDataFrameToCamelCase(pd.read_csv("./datasets/winequality.csv"))
dataset_numcols = dataset.select_dtypes(include=[float, int])
dataset_numcols.drop(["good", "quality"], axis=1, inplace=True)

scaled_dataset = pd.DataFrame(StandardScaler().fit_transform(dataset_numcols), columns=dataset_numcols.columns)

scaled_dataset_for_predict = pd.concat([scaled_dataset, dataset["color"]], axis=1)

X = scaled_dataset_for_predict.drop("color", axis=1)
y = scaled_dataset_for_predict["color"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

decision_tree_model = DecisionTreeClassifier(criterion="gini", random_state=42)
decision_tree_model.fit(X_train, y_train)

print(decision_tree_model.get_depth())
print(decision_tree_model.get_n_leaves())
#decision_tree_predictions = decision_tree_model.predict(X_test)

#print(classification_report(y_test, decision_tree_predictions))

#decision_tree_confusion_matrix = confusion_matrix(y_test, decision_tree_predictions)

#disp = ConfusionMatrixDisplay(confusion_matrix=decision_tree_confusion_matrix,
#                              display_labels=decision_tree_model.classes_)
#disp.plot()

#dot_data = tree.export_graphviz(decision_tree_model, out_file=None, feature_names=X.columns,
#                                class_names=decision_tree_model.classes_, filled=True, rounded=True,
#                                special_characters=True)

#graph = graphviz.Source(dot_data)
#graph.view()


def load_test_data():
    return dataset.to_dict(orient="records")


def predict_by_tree_model(data):
    test_datatest = pd.DataFrame(data.dict()["wines"])
    numeric_dataset = test_datatest.select_dtypes(include=[float, int])
    scalar_dataset = pd.DataFrame(StandardScaler().fit_transform(numeric_dataset), columns=numeric_dataset.columns)
    data_predictions = decision_tree_model.predict(scalar_dataset)
    print(classification_report(test_datatest["color"], data_predictions))
    test_datatest["color"] = data_predictions
    return test_datatest.to_dict(orient="records")
