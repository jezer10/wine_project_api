from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from prediction_models import predict_by_tree_model, predict_by_svm, predict_by_random_forest
from analysis_secuence import graph_elbow, hierarchical_graph, scatter_graph, pca_graph
import uvicorn
from wine_dto import WinesDTO

app = FastAPI()
templates = Jinja2Templates(directory="templates",
                            variable_start_string='[[',
                            variable_end_string=']]')


@app.get('/')
async def greetings():
    return "Hello World!"


@app.get('/graph/pca')
async def generate_scatter_graph():
    return pca_graph()


@app.get('/graph/scatter')
async def generate_scatter_graph():
    return scatter_graph()


@app.get('/graph/hierarchical')
async def generate_hierarchical_graph():
    return hierarchical_graph()
@app.get('/graph/elbow')
async def generate_elbow_graph():
    return graph_elbow()
@app.get("/predict/svm")
async def predict_svm(wines: WinesDTO):
    return predict_by_svm(wines)


@app.get("/predict/tree")
async def predict_tree(wines: WinesDTO):
    return predict_by_tree_model(wines)


@app.get("/predict/forest")
async def predict_tree(wines: WinesDTO):
    return predict_by_random_forest(wines)



if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info", host="0.0.0.0")
