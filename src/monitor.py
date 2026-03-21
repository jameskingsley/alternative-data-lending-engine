import pandas as pd
import os
import warnings
from clearml import Task

# Modern 0.7.21 Imports
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

# Ignore division warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_drift_monitoring():
    task = Task.init(
        project_name='Micro-Lending Engine',
        task_name='Data Drift Analysis',
        task_type=Task.TaskTypes.monitor
    )

    print("Loading data for drift analysis...")
    df = pd.read_csv('data/processed/train_final.csv')
    
    # 
    # I identify numerical and categorical columns for Evidently
    numeric_feats = df.select_dtypes(include=['number']).columns.tolist()
    # Remove IDs and Target from the features list
    for col in ['TARGET', 'SK_ID_CURR']:
        if col in numeric_feats:
            numeric_feats.remove(col)
            
    schema = DataDefinition(numerical_columns=numeric_feats)

    #  Create Evidently Datasets
    # Reference (Historical/Train) vs Current (Production/Test)
    reference_df = df.sample(frac=0.5, random_state=42)
    current_df = df.sample(frac=0.5, random_state=7)

    eval_ref = Dataset.from_pandas(reference_df, data_definition=schema)
    eval_curr = Dataset.from_pandas(current_df, data_definition=schema)

    print(" Running Evidently Evaluation...")
    #  Get a Report Template
    report = Report([DataDriftPreset()])
    
    #  Capture the Result Object
    my_eval = report.run(eval_curr, eval_ref)

    #  Save the results locally
    os.makedirs('models', exist_ok=True)
    json_path = 'models/drift_baseline.json'
    html_path = 'models/drift_report.html'
    
    print("Exporting results...")
    # These methods are now called on the result object (my_eval)
    my_eval.save_json(json_path)
    my_eval.save_html(html_path)

    # Upload to ClearML
    task.upload_artifact(name='drift_json', artifact_object=json_path)
    task.upload_artifact(name='drift_html', artifact_object=html_path)
    
    print(f" Success! Report saved to {html_path}")
    print(f" Monitor Task complete. ID: {task.id}")

if __name__ == "__main__":
    run_drift_monitoring()