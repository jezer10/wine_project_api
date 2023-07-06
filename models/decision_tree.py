# USO DE METODO DE AGRUPACION PARA LA PREDICCION DE VARIABLES
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pandas as pd

dataset_wine = pd.read_csv("data/wine_quality.csv")




def graph_model():
    dot_data = export_graphviz(model, out_file=None, feature_names=X.columns,
                                    class_names=model.classes_, filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.view()

