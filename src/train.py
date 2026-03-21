import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

# MLOps Imports
from clearml import Task, OutputModel

# Ignore standard runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize ClearML
task = Task.init(
    project_name='Micro-Lending Engine',
    task_name='Model Competition: Production Registry',
    output_uri=True 
)

def train_and_compare():
    # Load Data
    print("Loading data...")
    if not os.path.exists('data/processed/train_final.csv'):
        print("Error: data/processed/train_final.csv not found!")
        return

    df = pd.read_csv('data/processed/train_final.csv')
    X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Models
    models = {
        "Logistic_Regression_Scaled": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs'))
        ]),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            random_state=42,
            verbose=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            early_stopping_rounds=50, 
            random_state=42
        )
    }

    best_auc = 0
    best_model = None
    best_model_name = ""

    # Training Loop
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        try:
            if name == "LightGBM":
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_test, y_test)], 
                    callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )
            elif name == "XGBoost":
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                model.fit(X_train, y_train)

            # Evaluate
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            print(f"Result for {name}: AUC = {auc:.4f}")
            
            task.get_logger().report_single_value(name=f'{name}_AUC', value=auc)

            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f" Error training {name}: {e}")

    #  Finalize the Winner
    print(f"\n Overall Winner: {best_model_name} with AUC: {best_auc:.4f}")
    
    # Save Locally
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_lending_model.pkl'
    joblib.dump(best_model, model_path)

    # --- Register in Model Tab ---
    print(f" Uploading {best_model_name} to ClearML Models Tab...")
    output_model = OutputModel(task=task, name="Lending-Engine-Winner")
    output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
    
    print(f"Registered {best_model_name} with ID: {output_model.id}")
    print("Model successfully uploaded. ")

if __name__ == "__main__":
    train_and_compare()