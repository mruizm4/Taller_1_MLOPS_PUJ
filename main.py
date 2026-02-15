from fastapi import FastAPI, Form, Query
import pandas as pd
from typing import List, Annotated
from pydantic import BaseModel, Field
from predict import predict_new_data, load_model
from enum import Enum


app = FastAPI()


tree_model, tree_encoders, tree_scaler = load_model("decision_tree.pkl")
knn_model, knn_encoders, knn_scaler = load_model("knn.pkl")
svm_model, svm_encoders, svm_scaler = load_model("svm.pkl")

models_dict = {
    "TREE":{"model": tree_model, "encoders":tree_encoders, "scaler":tree_scaler },
    "KNN":{"model": knn_model, "encoders":knn_encoders, "scaler":knn_scaler },
    "SVM":{"model": svm_model, "encoders":svm_encoders, "scaler":svm_scaler }
}

class islas_class(str, Enum):
    Torgersen = "Torgersen"
    Dream = "Dream"
    Biscoe = "Biscoe"


class sex_class(str, Enum):
    Male = "Male"
    Female = "Female"

class model_class(str, Enum):
    TREE = "TREE"
    KNN = "KNN"
    SVM = "SVM"



@app.post("/predict")
async def root(
    models: Annotated[List[model_class], Query(..., description = "List of models: Decision Tree, K-Nearest Neighbors, Support Vector Machines")],
    culmen_length_mm: float = Query(39, description = "culmen length in millimeters (mm)"),
    culmen_depth_mm: float = Query(18.7, description = "culmen depth in millimeters (mm)"),
    flipper_length_mm: float = Query(180, description = " flipper length in millimeters (mm)"),
    body_mass_g: float = Query(3700, description = "body mass (g)"),
    island: islas_class = Query(islas_class.Torgersen, description = "island to which it belongs"),
    sex: sex_class = Query(sex_class.Male, description = "penguin gender"),
):


    df = pd.DataFrame([{
        "culmen_length_mm": culmen_length_mm,
        "culmen_depth_mm": culmen_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "island": island,
        "sex": sex
    }])

    response = {}

    for m in models:
        model_name = m.value

        prediction = predict_new_data(
            df,
            models_dict[model_name]["model"],
            models_dict[model_name]["encoders"],
            models_dict[model_name]["scaler"]
        )

        response[model_name] = prediction.tolist()

    return response