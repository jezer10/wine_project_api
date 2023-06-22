from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from prediction_tree_model import predict_by_tree_model, load_test_data
from pydantic import BaseModel
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates",
                            variable_start_string='[[',
                            variable_end_string=']]')


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
    color: Optional[str]


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


if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, log_level="info")
