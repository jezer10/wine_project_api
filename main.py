from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from prediction_models import predict_by_tree_model, predict_by_svm, predict_by_random_forest
from pydantic import BaseModel
import uvicorn
from wine_dto import WinesDTO

app = FastAPI()
templates = Jinja2Templates(directory="templates",
                            variable_start_string='[[',
                            variable_end_string=']]')


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
    uvicorn.run("main:app", port=5000, log_level="info")
