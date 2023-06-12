from typing import List
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from prediction_tree_model import predict_by_tree_model, load_test_data
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "nombre": "Juan"})


class WineDTO(BaseModel):
    fixedAcidity: float
    volatileAcidity: float
    citricAcid: float
    residualSugar: float
    chlorides: float
    freeSulfurDioxide: float
    totalSulfurDioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    color: str


class WinesDTO(BaseModel):
    wines: List[WineDTO]


@app.get("/load")
async def predict(request: Request):
    return {
        "data": load_test_data()
    }


@app.get("/predict")
async def predict_dto(wines: WinesDTO):
    return predict_by_tree_model(wines)


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
