# Alternative Data Lending Engine

An end-to-end Machine Learning pipeline designed for Real-time Credit Risk Assessment in emerging markets. This engine utilizes alternative behavioral markers and live macroeconomic data (Nigeria Inflation via World Bank API) to provide transparent, risk-adjusted default probabilities.

###### Live Production Links
Frontend UI: (https://alternative-data-lending-engine-hyfvawe49kajnnyvtjgcfa.streamlit.app/) — Interactive assessment & SHAP explainability.

Backend API: https://alternative-data-lending-engine.onrender.com — High-performance inference engine.

Observability: https://app.evidently.cloud/v2/projects/019d0fee-0a7f-72dd-9eaa-a360e6cf8a42/dashboard — Live Data Drift & Model Performance monitoring.

###### Architecture & MLOps Maturity
Modeling: Hybrid ensemble (XGBoost & LightGBM) optimized for imbalanced credit datasets.

Explainability (XAI): Integrated SHAP values to provide transparency for every automated decision (e.g., why an applicant was rejected).

Macro-Economic Integration: Live ingestion of Nigeria Inflation Data (33.24%) via World Bank API to adjust risk thresholds dynamically.

Orchestration & Tracking: ClearML for experiment tracking and model versioning; Prefect for ETL pipeline automation.

Monitoring: Evidently AI tracks feature drift on critical markers like DAYS_REGISTRATION and EXT_SOURCE_1 to ensure model reliability over time.

###### Key Indicators Analyzed
Behavioral Proxies: Analyzes HOUR_APPR_PROCESS_START and DAYS_REGISTRATION as proxies for applicant stability.

External Anchors: Incorporates normalized scores (EXT_SOURCE_1) to balance alternative data with traditional indicators.

Real-time Macro Context: Automatically weights risk higher during periods of high inflation to protect the lending portfolio.

###### Deployment Info
The system is containerized and deployed using a split-architecture:

Inference: FastAPI hosted on Render for low-latency scoring.

UI: Streamlit Cloud for executive-level visualization and model interpretability.

Version: Lending-Engine-Winner (Current Production Tag).