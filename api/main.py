import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from clearml import Task, Model 
from src.data_processor import DataProcessor

app = FastAPI(title="Lending Inference API")

# Initialize ClearML Task
Task.init(
    project_name='Micro-Lending Engine', 
    task_name='Inference Service', 
    task_type=Task.TaskTypes.inference, 
    reuse_last_task_id=True
)

PROJECT_NAME = 'Micro-Lending Engine'
MODEL_NAME = 'Lending-Engine-Winner'

def load_registry_model():
    print(f" Querying ClearML Production Registry for '{MODEL_NAME}'...")
    
    try:
        # Most direct way to get a single specific model by name/project
        best_model_obj = Model.get_model(
            project_name=PROJECT_NAME,
            model_name=MODEL_NAME,
            only_published=True
        )
    except Exception:
        # Fallback to query_models if get_model fails
        models = Model.query_models(
            project_name=PROJECT_NAME,
            model_name=MODEL_NAME,
            only_published=True
        )
        if not models:
            raise RuntimeError(f"FATAL: No Published model '{MODEL_NAME}' found.")
        
        # Robust sort using the internal configuration timestamp
        best_model_obj = sorted(models, key=lambda x: x.last_update if hasattr(x, 'last_update') else 0, reverse=True)[0]

    print(f" Downloading Model ID: {best_model_obj.id}")
    model_path = best_model_obj.get_local_copy()
    
    # Load and setup SHAP
    loaded_model = joblib.load(model_path)
    loaded_explainer = shap.TreeExplainer(loaded_model)
    
    print(f"Success! Connection established.")
    return loaded_model, loaded_explainer

# Global model initialization
model, explainer = load_registry_model()
processor = DataProcessor()

class BorrowerData(BaseModel):
    features: dict 

@app.post("/predict")
def predict(data: BorrowerData):
    try:
        input_df = pd.DataFrame([data.features])
        
        # Real-time Macro Data Enrichment
        macro_df = processor.fetch_world_bank_data()
        input_df['macro_inflation'] = macro_df.loc[macro_df['series'] == 'FP.CPI.TOTL.ZG', 'YR2024'].values[0]
        input_df['macro_gdp'] = macro_df.loc[macro_df['series'] == 'NY.GDP.MKTP.KD.ZG', 'YR2024'].values[0]

        # Column Alignment
        if hasattr(model, "get_booster"):
            expected_cols = model.get_booster().feature_names
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)
        elif hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        prob = model.predict_proba(input_df)[:, 1][0]
        
        # Explainability
        shap_values = explainer.shap_values(input_df)
        risk_factors = pd.Series(
            shap_values[0], 
            index=input_df.columns
        ).sort_values(ascending=False).head(3).to_dict()

        return {
            "probability": round(float(prob), 4),
            "decision": "APPROVED" if prob < 0.12 else "REJECTED",
            "risk_factors": risk_factors,
            "metadata": {
                "registry_name": MODEL_NAME,
                "inflation_rate": f"{input_df['macro_inflation'].values[0]:.2%}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))