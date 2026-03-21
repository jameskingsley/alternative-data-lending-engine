from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from clearml import Task, Model 
from src.data_processor import DataProcessor

app = FastAPI(title="Lending Inference API")

# Initialize ClearML Task for the Inference Service
Task.init(
    project_name='Micro-Lending Engine', 
    task_name='Inference Service', 
    task_type=Task.TaskTypes.inference, 
    reuse_last_task_id=True
)

try:
    print("📡 Querying ClearML Production Registry for 'Lending-Engine-Winner'...")
    
    # This search finds the latest model based on the Name and Project
    # 'only_published=True' ensures you only get the model you officially vetted
    best_model_obj = Model.get_model(
        project_name='Micro-Lending Engine',
        model_name='Lending-Engine-Winner',
        only_published=True
    )

    if not best_model_obj:
        raise ValueError("No 'Published' model found with name 'Lending-Engine-Winner'.")

    # Download to local cache and load into memory
    model_path = best_model_obj.get_local_copy()
    model = joblib.load(model_path)
    
    # Initialize SHAP for explainability
    explainer = shap.TreeExplainer(model)
    print(f"✅ Success! Loaded Model ID: {best_model_obj.id} (Name: {best_model_obj.name})")
    
except Exception as e:
    print(f"⚠️ ClearML Registry Error: {e}")
    # Local fallback to keep the service running in Enugu
    try:
        print("🔄 Attempting local fallback to 'models/best_lending_model.pkl'...")
        model = joblib.load("models/best_lending_model.pkl")
        explainer = shap.TreeExplainer(model)
        print("✅ Local fallback successful.")
    except Exception as fallback_err:
        raise RuntimeError(f"FATAL: Could not load model from ClearML or Local: {fallback_err}")

processor = DataProcessor()

class BorrowerData(BaseModel):
    features: dict 

@app.post("/predict")
def predict(data: BorrowerData):
    try:
        # Create DataFrame from JSON
        input_df = pd.DataFrame([data.features])
        
        # Real-time Macro Data Enrichment
        macro_df = processor.fetch_world_bank_data()
        input_df['macro_inflation'] = macro_df.loc[macro_df['series'] == 'FP.CPI.TOTL.ZG', 'YR2024'].values[0]
        input_df['macro_gdp'] = macro_df.loc[macro_df['series'] == 'NY.GDP.MKTP.KD.ZG', 'YR2024'].values[0]

        # Handle Column Alignment for XGBoost
        if hasattr(model, "get_booster"):
            expected_cols = model.get_booster().feature_names
            input_df = input_df.reindex(columns=expected_cols, fill_value=0)
        
        # Generate Probability & Decision
        prob = model.predict_proba(input_df)[:, 1][0]
        
        #  Explainability (SHAP Values)
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
                "registry_name": "Lending-Engine-Winner",
                "inflation_rate": f"{input_df['macro_inflation'].values[0]:.2%}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))