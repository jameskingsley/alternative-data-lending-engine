import streamlit as st
import requests

# Configuration - Point this to the LIVE Render URL
API_URL = "https://alternative-data-lending-engine.onrender.com/predict"

st.set_page_config(page_title="Nigerian Micro-Lending Portal", layout="wide", page_icon="💰")

st.title(" Alternative Data Lending Engine")
st.subheader("Real-time Credit Assessment (Production)")

with st.sidebar:
    st.header("Applicant Profile")
    # Financial Basics - Matching the Model's Feature Names
    income = st.number_input("Monthly Income (NGN Total)", value=150000, step=10000)
    credit = st.number_input("Requested Loan Amount", value=500000, step=10000)
    goods_price = st.number_input("Goods Price (if applicable)", value=450000)
    
    # Stability & Demographics
    employed_days = st.number_input("Days Employed (Negative integer)", value=-1000)
    id_publish_days = st.number_input("Days Since ID Published (Negative)", value=-2000)
    birth_days = st.number_input("Days Since Birth (Negative, e.g. -12000 for ~33yrs)", value=-12000)
    children = st.slider("Number of Children", 0, 10, 2)
    
    st.markdown("---")
    st.header(" External & Macro Data")
    # High-impact feature from the logs
    ext_source_2 = st.slider("External Credit Score (0.0 to 1.0)", 0.0, 1.0, 0.5)
    
    st.info("The API fetches real World Bank data automatically.")

if st.button("Assess Loan Risk"):
    # Constructing payload with EXACT keys expected by the XGBoost booster
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_BIRTH": birth_days,
            "DAYS_EMPLOYED": employed_days,
            "DAYS_ID_PUBLISH": id_publish_days,
            "CNT_CHILDREN": children,
            "EXT_SOURCE_2": ext_source_2
        }
    }
    
    with st.spinner(" Querying Inference Engine & World Bank API..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status() # Check for HTTP errors
            result = response.json()
            
            # Layout for Results
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                prob = result['probability']
                st.metric("Default Probability", f"{prob:.2%}")
                
                # Visual Risk Meter
                if result['decision'] == "APPROVED":
                    st.success(f" Decision: {result['decision']}")
                    st.balloons()
                else:
                    st.error(f" Decision: {result['decision']}")
                
                # Simple progress bar as a risk gauge
                st.write("**Risk Level:**")
                st.progress(prob)
            
            with col2:
                st.write("### Risk Breakdown (SHAP)")
                for factor, impact in result['risk_factors'].items():
                    # Color code the impact for the loan officer
                    if impact > 0:
                        st.write(f" **{factor}**: +{impact:.4f} (Increases Risk)")
                    else:
                        st.write(f" **{factor}**: {impact:.4f} (Mitigates Risk)")
            
            with col3:
                st.write("###  Contextual Metadata")
                st.write(f"**Model:** `{result['metadata'].get('registry_name')}`")
                st.write(f"**Registry:** ClearML Production")
                
                # Dynamic inflation display from the API metadata
                inf = result['metadata'].get('inflation_rate', 'N/A')
                st.write(f"**Nigeria Inflation (Real-time):** `{inf}`")
                st.caption("Data source: World Bank API")

        except Exception as e:
            st.error(f"API Connection Failed.")
            st.info(f"Check if the Render service is awake at: {API_URL}")
            st.error(f"Error Details: {e}")