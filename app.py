import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean, enterprise-grade page configuration
st.set_page_config(page_title="Loan Intelligence Portal", layout="wide", page_icon="🏦")

# Custom CSS injection for premium card styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOAD & CACHE DATA ASSETS
# ==========================================
@st.cache_resource
def load_banking_assets():
    return joblib.load('loan_approval_assets.pkl')

assets = load_banking_assets()
log_reg = assets['logistic_regression']
rf_model = assets['random_forest']
kmeans = assets['kmeans']
pca = assets['pca']
scaler = assets['scaler']
label_encoders = assets['label_encoders']
cluster_plot_data = assets['cluster_plot_data']
segment_profiles = assets['segment_profiles']

# ==========================================
# 2. UI HEADER LAYOUT
# ==========================================
st.title("🏦 Next-Gen Credit Decisions & Behavioral Analytics")
st.markdown("An end-to-end multi-model risk intelligence engine tailored for retail banking evaluation.")
st.markdown("---")

# ==========================================
# 3. INTERACTIVE SIDEBAR INPUT CONTROLS
# ==========================================
st.sidebar.header("📋 Applicant Financial Profile")

# Layout parameters partitioned by logic
with st.sidebar.expander("💳 Demographics & Core Credit", expanded=True):
    age = st.number_input("Age", min_value=18, max_value=100, value=34)
    credit_score = st.slider("Credit Score", 300, 850, 710)
    education = st.selectbox("Education Level", label_encoders['Education'].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders['MaritalStatus'].classes_)

with st.sidebar.expander("💼 Employment & Income", expanded=True):
    income = st.number_input("Annual Income ($)", min_value=0, value=72000, step=5000)
    months_employed = st.number_input("Months Employed", min_value=0, value=60)
    employment_type = st.selectbox("Employment Type", label_encoders['EmploymentType'].classes_)

with st.sidebar.expander("💰 Loan Request Specifics", expanded=True):
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=25000, step=1000)
    interest_rate = st.slider("Interest Rate (%)", 1.0, 35.0, 10.5, step=0.1)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
    loan_purpose = st.selectbox("Loan Purpose", label_encoders['LoanPurpose'].classes_)
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.28, step=0.01)

with st.sidebar.expander("🏠 Financial Liabilities", expanded=False):
    has_mortgage = st.selectbox("Has Active Mortgage?", label_encoders['HasMortgage'].classes_)
    has_dependents = st.selectbox("Has Dependents?", label_encoders['HasDependents'].classes_)
    has_co_signer = st.selectbox("Has Co-Signer?", label_encoders['HasCoSigner'].classes_)
    num_credit_lines = st.slider("Open Credit Lines", 1, 20, 4)

# ==========================================
# 4. INFERENCE PROCESSING PIPELINE
# ==========================================
input_df = pd.DataFrame({
    'Age': [age], 'Income': [income], 'LoanAmount': [loan_amount], 'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed], 'NumCreditLines': [num_credit_lines], 'InterestRate': [interest_rate],
    'LoanTerm': [loan_term], 'DTIRatio': [dti_ratio], 'Education': [education], 'EmploymentType': [employment_type],
    'MaritalStatus': [marital_status], 'HasMortgage': [has_mortgage], 'HasDependents': [has_dependents],
    'LoanPurpose': [loan_purpose], 'HasCoSigner': [has_co_signer]
})

# Feature transformation maps matching training environments
encoded_input = input_df.copy()
for col in label_encoders.keys():
    encoded_input[col] = label_encoders[col].transform(encoded_input[col].astype(str))

scaled_input = scaler.transform(encoded_input)
pca_input = pca.transform(scaled_input)

# Calculate live outputs across all models
lr_decision = log_reg.predict(scaled_input)[0]
rf_risk_prob = rf_model.predict_proba(encoded_input)[0][1]
assigned_cluster = kmeans.predict(scaled_input)[0]

# ==========================================
# 5. DASHBOARD MAIN INTERFACE TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Core Credit Decisions (LR & RF)", 
    "👥 Behavioral Customer Clustering (K-Means)", 
    "🧩 Latent Vector Space (PCA)"
])

# ------------------------------------------
# TAB 1: LOGISTIC REGRESSION & RANDOM FOREST
# ------------------------------------------
with tab1:
    st.subheader("Automated Underwriting Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Model 1: Gateway Decision (Logistic Regression)")
        if lr_decision == 0:
            st.success("✅ **STATUS: APPROVED**")
            st.caption("Applicant parameters align completely within safe variance bounds.")
        else:
            st.error("❌ **STATUS: REJECTED**")
            st.caption("High system default indicators triggered. Standard manual rejection advised.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Model 2: Risk Profile (Random Forest)")
        st.metric(label="Calculated Default Probability", value=f"{rf_risk_prob * 100:.2f}%")
        
        if rf_risk_prob < 0.25:
            st.info("🟢 **Tier status: Low Risk Profile**")
        elif rf_risk_prob < 0.55:
            st.warning("🟡 **Tier status: Borderline Profile (Requires Co-signer validation)**")
        else:
            st.error("🔴 **Tier status: Critically High Risk Exposure**")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: CLUSTERING VISUALIZATION & ANALYTICS
# ------------------------------------------
with tab2:
    st.subheader("Interactive Behavioral Segmentation Map")
    st.write(f"Based on macro-financial attributes, this applicant maps to **Cluster Segment {assigned_cluster}**.")

    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Customer Cohort Mapping (PCA Dimension Space)")
        st.caption("The graph below shows 1,000 sampled bank customers reduced to 2 Principal Components. The glowing red star represents your current applicant's real-time positioning inside the market cohorts.")
        
        # Build Seaborn Matplotlib Visual Chart
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(
            data=cluster_plot_data, 
            x='PC1', y='PC2', 
            hue='Cluster', 
            palette='Set2', 
            alpha=0.5, 
            ax=ax,
            hue_order=['0', '1', '2', '3']
        )
        
        # Superimpose current user coordinates live
        ax.scatter(
            pca_input[0][0], pca_input[0][1], 
            color='red', marker='*', s=350, edgecolor='black', linewidth=1.5,
            label='Current Applicant Profile'
        )
        
        ax.set_title("Customer Clusters View (PC1 vs PC2)", fontsize=10, weight='bold')
        ax.legend(facecolor='white', framealpha=1)
        sns.despine()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Structural Cohort Benchmark Profiles")
        st.caption("Compare key criteria averages for each distinct customer segment cluster.")
        
        # Generate clean stream chart metrics profile
        chart_metric = st.selectbox("Select Benchmark Metric", ["Income", "CreditScore", "LoanAmount"])
        
        fig_bar, ax_bar = plt.subplots(figsize=(5, 3.8))
        colors = ['#aec6cf' if i != assigned_cluster else '#ff6961' for i in range(4)]
        
        sns.barplot(
            x=segment_profiles.index, 
            y=segment_profiles[chart_metric], 
            palette=colors, 
            ax=ax_bar
        )
        
        ax_bar.set_title(f"Mean Portfolio {chart_metric} By Segment", fontsize=10, weight='bold')
        ax_bar.set_xlabel("Customer Segment ID")
        ax_bar.set_ylabel(chart_metric)
        
        # Line indicator showing user's current choice value vs clusters
        user_val = input_df[chart_metric].values[0]
        ax_bar.axhline(user_val, color='red', linestyle='--', alpha=0.8, label=f"Applicant ({user_val:,.0f})")
        ax_bar.legend(loc='upper right')
        
        st.pyplot(fig_bar)

# ------------------------------------------
# TAB 3: PCA DIMENSIONALITY MATRIX
# ------------------------------------------
with tab3:
    st.subheader("Model 4: Structural Features Compression (PCA Space)")
    st.markdown(f"The 16 high-dimensional features from the input profile have been compressed down into **{pca_input.shape[1]} core orthogonal directions** capturing 90% total system variance.")
    
    # Render dynamic data frame coordinates matrix
    coord_cols = [f"Principal Component {i+1}" for i in range(pca_input.shape[1])]
    pca_coordinates_df = pd.DataFrame(pca_input, columns=coord_cols)
    
    st.dataframe(pca_coordinates_df.style.format("{:.4f}"))