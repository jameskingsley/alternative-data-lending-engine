import streamlit as st
import requests

st.set_page_config(page_title="Nigerian Micro-Lending Portal", layout="wide")

st.title(" Alternative Data Lending Engine")
st.subheader("Real-time Credit Assessment (Production)")

with st.sidebar:
    st.header(" Applicant Profile")
    # Financial Basics
    income = st.number_input("Monthly Income (NGN)", value=150000, step=10000)
    credit = st.number_input("Requested Loan Amount", value=500000, step=10000)
    goods_price = st.number_input("Goods Price (if applicable)", value=450000)
    
    # Stability & Demographics
    employed_days = st.number_input("Days Employed (Negative)", value=-1000)
    id_publish_days = st.number_input("Days Since ID Published", value=-2000)
    children = st.slider("Number of Children", 0, 10, 2)
    
    st.markdown("---")
    st.header("External & Macro Data")
    # Capturing the high-impact feature seen in your SHAP logs
    ext_source_2 = st.slider("External Credit Score (0-1)", 0.0, 1.0, 0.5)
    
    # What-If Macro Simulation
    st.info("The API fetches real World Bank data, but you can simulate a shock below.")
    inflation_shock = st.slider("Simulate Inflation Increase (%)", 0, 50, 0)

if st.button("Assess Loan Risk"):
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_EMPLOYED": employed_days,
            "DAYS_ID_PUBLISH": id_publish_days,
            "CNT_CHILDREN": children,
            "EXT_SOURCE_2": ext_source_2,
            "inflation_shock": inflation_shock  
        }
    }
    
    with st.spinner(" Querying Inference Engine & World Bank API..."):
        try:
            # Pointing to your FastAPI port
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()
            
            # Layout for Results
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Default Probability", f"{result['probability']:.2%}")
                if result['decision'] == "APPROVED":
                    st.success(f"Decision: {result['decision']}")
                else:
                    st.error(f"Decision: {result['decision']}")
            
            with col2:
                st.write("### Risk Breakdown (SHAP)")
                for factor, impact in result['risk_factors'].items():
                    # Color code the impact for the loan officer
                    if impact > 0:
                        st.write(f" **{factor}**: {impact:.4f} (Increases Risk)")
                    else:
                        st.write(f" **{factor}**: {impact:.4f} (Decreases Risk)")
            
            with col3:
                st.write("###  Contextual Metadata")
                st.write(f"**Model Name:** Lending-Engine-Winner")
                st.write(f"**Source:** ClearML Production Registry")
                if "metadata" in result:
                    st.write(f"**Economic Context:** {result['metadata'].get('macro_used', '2024 Base')}")

        except Exception as e:
            st.error(f" API Connection Failed. Ensure FastAPI is running on port 8000.")
            st.info(f"Error Details: {e}")