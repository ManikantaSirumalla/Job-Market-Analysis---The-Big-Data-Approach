import os
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helpers
@st.cache_data(show_spinner=False, ttl=300)
def load_unified_salaries() -> pd.DataFrame:
    """Load salary data with better error handling"""
    try:
        silver_dir = Path("data/silver/unified_salaries")
        parquet_file = Path("data/silver/unified_salaries.parquet")
        
        if silver_dir.exists():
            df = pd.read_parquet(silver_dir)
            if len(df) > 0:
                return df
        
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            if len(df) > 0:
                return df
                
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading salary data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=300)
def get_gold_data(dataset_name: str) -> pd.DataFrame:
    """Safely load gold layer data"""
    try:
        gold_path = Path(f"data/gold/{dataset_name}.parquet")
        if gold_path.exists():
            return pd.read_parquet(gold_path)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=60)
def call_api_predict(years_experience: int, currency: str = "USD", period: str = "YEARLY"):
    """Call API for salary prediction with caching"""
    try:
        payload = {
            "years_experience": years_experience,
            "skills": [],
            "currency": currency,
            "period": period,
        }
        r = requests.post("http://localhost:8000/predict-salary", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"predicted_salary": None, "error": "API server not running. Please start with 'make api'"}
    except requests.exceptions.Timeout:
        return {"predicted_salary": None, "error": "API request timed out"}
    except Exception as e:
        return {"predicted_salary": None, "error": str(e)}

# Main app
st.title("💼 Job Market Analytics Dashboard")

# Sidebar navigation
section = st.sidebar.selectbox(
    "Navigate",
    ["KPIs Dashboard", "Salary Intelligence", "Salary Prediction", "Trends", "About"]
)

if section == "KPIs Dashboard":
    st.title("📊 KPIs Dashboard")
    
    # Load data
    df_salaries = load_unified_salaries()
    
    if df_salaries.empty:
        st.warning("No salary data available. Run the ETL first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df_salaries):,}")
        
        with col2:
            if "salary_amount" in df_salaries.columns:
                # Convert salary_amount to numeric, handling any string values
                df_salaries["salary_amount"] = pd.to_numeric(df_salaries["salary_amount"], errors='coerce')
                median_salary = df_salaries["salary_amount"].median()
                if pd.isna(median_salary):
                    st.metric("Median Salary", "N/A")
                else:
                    st.metric("Median Salary", f"${median_salary:,.0f}")
            else:
                st.metric("Median Salary", "N/A")
        
        with col3:
            if "currency" in df_salaries.columns:
                unique_currencies = df_salaries["currency"].nunique()
                st.metric("Currencies", unique_currencies)
            else:
                st.metric("Currencies", "N/A")
        
        with col4:
            if "period" in df_salaries.columns:
                unique_periods = df_salaries["period"].nunique()
                st.metric("Pay Periods", unique_periods)
            else:
                st.metric("Pay Periods", "N/A")

elif section == "Salary Intelligence":
    st.title("💰 Salary Intelligence")
    
    df_salaries = load_unified_salaries()
    
    if df_salaries.empty or "salary_amount" not in df_salaries.columns:
        st.warning("No salary data available. Run the ETL first.")
    else:
        # Convert salary_amount to numeric, handling any string values
        df_salaries["salary_amount"] = pd.to_numeric(df_salaries["salary_amount"], errors='coerce')
        # Remove any rows where salary conversion failed
        df_salaries = df_salaries.dropna(subset=['salary_amount'])
        
        if df_salaries.empty:
            st.warning("No valid salary data after conversion.")
        else:
            left, right = st.columns([2, 1])
            
            with left:
                st.subheader("Salary Distribution")
                fig = px.histogram(
                    df_salaries,
                    x="salary_amount",
                    nbins=50,
                    title="Distribution of Salaries",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with right:
                st.subheader("By Currency")
                if "currency" in df_salaries.columns:
                    cur = df_salaries["currency"].value_counts().reset_index()
                    cur.columns = ["currency", "count"]
                    fig2 = px.bar(cur.head(10), x="currency", y="count", template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No currency data available")

elif section == "Salary Prediction":
    st.title("🔮 Salary Prediction")
    
    st.subheader("ML-Powered Salary Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_experience = st.slider("Years of Experience", 0, 20, 5)
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])
    
    with col2:
        period = st.selectbox("Pay Period", ["YEARLY", "MONTHLY", "HOURLY"])
        skills = st.multiselect("Skills (optional)", ["Python", "Java", "JavaScript", "SQL", "Machine Learning"])
    
    if st.button("Predict Salary", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = call_api_predict(years_experience, currency, period)
            
            if result.get("predicted_salary"):
                predicted = result["predicted_salary"]
                lower = result.get("lower_bound", predicted * 0.8)
                upper = result.get("upper_bound", predicted * 1.2)
                
                st.success(f"Predicted Salary: ${predicted:,.0f}")
                st.info(f"Range: ${lower:,.0f} - ${upper:,.0f}")
                
                if result.get("model_metrics"):
                    st.caption(f"Model metrics: {result['model_metrics']}")
            else:
                st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

elif section == "Trends":
    st.title("📈 Market Trends")
    
    # Load trending data
    trending_jobs = get_gold_data("trending_jobs")
    trending_skills = get_gold_data("trending_skills")
    job_postings_trends = get_gold_data("job_postings_trends")
    gold_so_langs = get_gold_data("so_languages")
    
    # Trending Job Titles
    if not trending_jobs.empty and {"title","count"}.issubset(trending_jobs.columns):
        st.subheader("🔥 Trending Job Titles")
        fig_jobs = px.bar(trending_jobs.head(10), x="title", y="count", template="plotly_white")
        fig_jobs.update_layout(xaxis_tickangle=-30, yaxis_title="Posting Count")
        st.plotly_chart(fig_jobs, use_container_width=True)
    
    # Trending Skills
    if not trending_skills.empty and {"skill","count"}.issubset(trending_skills.columns):
        st.subheader("⚡ Most Demanded Skills")
        fig_skills = px.bar(trending_skills.head(15), x="skill", y="count", template="plotly_white")
        fig_skills.update_layout(xaxis_tickangle=-30, yaxis_title="Mention Count")
        st.plotly_chart(fig_skills, use_container_width=True)
    
    # Job Postings Volume Trends
    if not job_postings_trends.empty and {"date","postings"}.issubset(job_postings_trends.columns):
        st.subheader("📊 Job Postings Volume (Last 30 Days)")
        fig_postings = px.line(job_postings_trends, x="date", y="postings", template="plotly_white")
        fig_postings.update_layout(xaxis_title="Date", yaxis_title="Number of Postings")
        st.plotly_chart(fig_postings, use_container_width=True)
    
    # StackOverflow Language Trends
    if not gold_so_langs.empty and {"year","language","count"}.issubset(gold_so_langs.columns):
        st.subheader("💻 StackOverflow: Top Languages by Year")
        year_sel = st.selectbox("Select Year", sorted(gold_so_langs["year"].dropna().unique()))
        subset = gold_so_langs[gold_so_langs["year"] == year_sel].head(10)
        fig_so = px.bar(subset, x="language", y="count", template="plotly_white")
        fig_so.update_layout(xaxis_tickangle=-30, yaxis_title="Developer Count")
        st.plotly_chart(fig_so, use_container_width=True)
    else:
        st.info("StackOverflow language data not available.")

elif section == "About":
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Job Market Analysis Dashboard
    
    This dashboard provides insights into job market trends, salary predictions, and career analytics.
    
    ### Data Sources:
    - **GitHub Archive**: Developer activity and repository trends
    - **StackOverflow Surveys**: Developer skills and preferences
    - **BLS Data**: Employment and wage statistics
    - **Kaggle Job Postings**: Current job market data
    
    ### Features:
    - 📊 **KPIs Dashboard**: Key metrics and statistics
    - 💰 **Salary Intelligence**: Interactive salary analysis
    - 🔮 **Salary Prediction**: ML-powered salary predictions
    - 📈 **Trends**: Market trends and skill demand
    - ℹ️ **About**: Project information
    
    ### Technology Stack:
    - **Backend**: FastAPI, Python
    - **Frontend**: Streamlit
    - **ML**: XGBoost, MLflow
    - **Data Processing**: Apache Spark, Delta Lake
    - **Orchestration**: Apache Airflow
    - **Streaming**: Apache Kafka
    """)

# Footer
st.markdown("---")
st.caption("Job Market Analysis Dashboard | Built with Streamlit")
