from models.load_dataset import X, X_train, y_train
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

print(
    f"Loaded Decisión Tree Model\n"
    f"Depth: {model.get_depth()}\n"
    f"Leaves N: {model.get_n_leaves()}"
)


def graph_model():
    dot_data = export_graphviz(model, out_file=None, feature_names=X.columns,
                                    class_names=model.classes_, filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()

