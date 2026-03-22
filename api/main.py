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

# Configuration for Model Loading
PROJECT_NAME = 'Micro-Lending Engine'
MODEL_NAME = 'Lending-Engine-Winner'

try:
    print(f"Querying ClearML Production Registry for '{MODEL_NAME}'...")
    
    # method to find published models
    models = Model.list_models(
        project_name=PROJECT_NAME,
        model_name=MODEL_NAME,
        only_published=True,
        order_by_field='created',
        ascending=False
    )

    if not models:
        raise ValueError(f"No 'Published' model found with name '{MODEL_NAME}'.")

    # Get the latest published model
    best_model_obj = models[0]
    model_path = best_model_obj.get_local_copy()
    model = joblib.load(model_path)
    
    explainer = shap.TreeExplainer(model)
    print(f"Success! Loaded Model ID: {best_model_obj.id}")
    
except Exception as e:
    print(f" ClearML Registry Error: {e}")
    try:
        print("Attempting local fallback...")
        # Resolve path dynamically to handle Render's directory structure
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(base_dir, "models", "best_lending_model.pkl")
        
        print(f"Looking for fallback at: {fallback_path}")
        model = joblib.load(fallback_path)
        explainer = shap.TreeExplainer(model)
        print("Local fallback successful.")
    except Exception as fallback_err:
        print(f" Fallback failed: {fallback_err}")
        raise RuntimeError(f"FATAL: Could not load model: {fallback_err}")

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

        # Column Alignment for XGBoost/Scikit-Learn
        if hasattr(model, "get_booster"):
            expected_cols = model.get_booster().feature_names
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)
        elif hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Prediction
        prob = model.predict_proba(input_df)[:, 1][0]
        
        # SHAP Explainability
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