import io
import pandas as pd
from models.load_dataset import dataset_numeric_columns
from fastapi import FastAPI, Request, Response, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from prediction_models import predict_by_tree_model, predict_by_svm, predict_by_random_forest
from analysis_secuence import graph_elbow, \
    hierarchical_graph, \
    scatter_graph, pca_graph, \
    silhouette_graph, \
    missing_matrix_graph, \
    histogram_graph, \
    barplot_graph, \
    get_dataset_columns
import uvicorn
from wine_dto import WinesDTO

from models.load_dataset import get_numeric_dataset, get_dataset_mean

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates",
                            variable_start_string='[[',
                            variable_end_string=']]')


@app.get("/summary")
async def get_summary():
    return {
        "all": get_numeric_dataset(),
        "white": get_numeric_dataset("white"),
        "red": get_numeric_dataset("red")
    }


@app.post('/upload')
async def upload_file(file: UploadFile):
    filename = file.filename
    file_extension = filename.split(".")[-1]
    if file_extension not in ['xlsx', 'xls', 'csv']:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido.")
    buffer = io.BytesIO(await file.read())
    df = (pd.read_csv(buffer) if file_extension == "csv" else pd.read_excel(buffer)).utilities.capitalize()

    if not all(c in dataset_numeric_columns.columns for c in df.columns):
        raise HTTPException(status_code=400, detail="Columnas Invalidas")
    return df.to_dict(orient="records")[0]


@app.get('/')
async def greetings():
    return "Hello World!"


@app.get('/profile')
async def get_dataset_profile():
    return get_dataset_mean()


@app.get("/graph/bar")
async def generate_bar_plot_graph(column: str):
    return barplot_graph(column)


@app.get("/graph/histogram")
async def generate_histogram_graph(column: str, response_class=Response):
    return histogram_graph(column)


@app.get("/graph/missing")
async def generate_missing_matrix_graph():
    return missing_matrix_graph()


@app.get('/graph/pca')
async def generate_pca_graph():
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


@app.get('/graph/silhouette')
async def generate_silhouette_graph():
    return silhouette_graph()


@app.post("/predict/svm")
async def predict_svm(wines: WinesDTO):
    return predict_by_svm(wines)


@app.post("/predict/tree")
async def predict_tree(wines: WinesDTO):
    return predict_by_tree_model(wines)


@app.post("/predict/forest")
async def predict_tree(wines: WinesDTO):
    return predict_by_random_forest(wines)


@app.get("/columns")
async def get_columns():
    return get_dataset_columns()


if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info", host="0.0.0.0")
