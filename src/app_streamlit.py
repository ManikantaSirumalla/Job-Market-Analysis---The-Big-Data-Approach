
import os
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Job Market Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
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
                
        # Return empty DataFrame if no data found
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading salary data: {e}")
        return pd.DataFrame()

def kpi_card(label: str, value: str, help_text: str = ""):
    st.metric(label=label, value=value, help=help_text)

@st.cache_data(show_spinner=False, ttl=60)  # Cache predictions for 1 minute
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

@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
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

@st.cache_data(show_spinner=False, ttl=60)  # Cache for 1 minute
def call_api_forecast(skill: str, horizon: int = 12):
    try:
        payload = {"skill": skill, "horizon_months": horizon}
        r = requests.post("http://localhost:8000/forecast-skills", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"forecast": []}

@st.cache_data(show_spinner=False, ttl=60)  # Cache for 1 minute
def call_api_career_recs(current_skills: list[str], target_role: str | None = None):
    try:
        payload = {"current_skills": current_skills, "target_role": target_role}
        r = requests.post("http://localhost:8001/career-recommendations", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"recommended_skills": [], "error": str(e)}

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("💼 Job Market Analytics")

section = st.sidebar.radio(
    "Navigate",
    [
        "Market Overview",
        "Skill Analysis", 
        "Salary Intelligence",
        "Career Planner",
        "Real-time Trends",
        "About",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("API: http://localhost:8000 | Dashboard: http://localhost:8501")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<style>
.kpi-card {background: #0f172a0d; padding: 16px; border-radius: 12px; border: 1px solid #e2e8f0;}
.kpi-title {font-size: 13px; color: #64748b; margin-bottom: 6px}
.kpi-value {font-size: 22px; font-weight: 600; color: #0f172a}
</style>
""", unsafe_allow_html=True)

# Load data lazily - only when needed
@st.cache_data(show_spinner=False, ttl=300)
def get_salaries_data():
    df = load_unified_salaries()
    if not df.empty and "salary_amount" in df:
        df["salary_amount"] = pd.to_numeric(df["salary_amount"], errors="coerce")
    return df

# Curated silver postings
@st.cache_data(show_spinner=False)
def load_unified_postings() -> pd.DataFrame:
    # prefer single parquet, fallback to directory
    p1 = Path("data/silver/unified_postings.parquet")
    p2 = Path("data/silver/unified_postings")
    try:
        if p1.exists():
            return pd.read_parquet(p1)
        if p2.exists():
            return pd.read_parquet(p2)
    except Exception:
        pass
    return pd.DataFrame()

# ------------------------------------------------------------
# Gold loaders (Parquet mirrors) - Lazy loading
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)  # Cache for 5 minutes
def load_gold_df(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        if p.exists():
            return pd.read_parquet(p)
    except Exception:
        pass
    return pd.DataFrame()

# Load data only when needed, not on startup
def get_gold_data(data_name: str) -> pd.DataFrame:
    """Lazy load gold data only when needed"""
    data_paths = {
        "ml_features": "data/gold/ml_features",
        "skills_demand": "data/gold/skills_demand.parquet",
        "location_hotspots": "data/gold/location_hotspots", 
        "so_top_languages": "data/gold/so_top_languages",
        "so_devtype_distribution": "data/gold/so_devtype_distribution",
        "gh_repo_type_counts": "data/gold/gh_repo_type_counts",
        "gh_hourly_trends": "data/gold/gh_hourly_trends",
        "gh_language_monthly": "data/gold/gh_language_monthly",
        "tech_job_linkage": "data/gold/tech_job_linkage",
        "company_hiring_velocity": "data/gold/company_hiring_velocity",
        "company_size_distribution": "data/gold/company_size_distribution",
        "bls_health_index": "data/gold/bls_health_index"
    }
    return load_gold_df(data_paths.get(data_name, ""))

# ------------------------------------------------------------
# Trends (job titles and skills)
# ------------------------------------------------------------
TITLE_NOISE = {
    "religion","racial","color","gender","disability","veteran","eeo","equal",
    "vision","dental","medical","401k","hourly","yearly","weekly","benefit","benefits",
    "drug","background","check","sign","bonus","insurance","policy","policies","pto",
}
TITLE_ALLOW_KEYWORDS = {
    "engineer","developer","scientist","analyst","manager","architect","lead","specialist",
    "consultant","product","program","project","qa","sdet","security","devops","cloud",
    "ml","ai","data","database","bi","business intelligence","backend","front end","frontend",
    "full stack","ios","android","mobile","platform","infra","site reliability","sre",
}
SKILL_NOISE = {
    "and","or","with","the","to","of","in","for","a","an","on","at","as","by",
    "benefits","vision","dental","medical","policy","policies","eeo","religion","color",
}

def is_real_job_title(title: str) -> bool:
    t = title.strip().lower()
    if len(t) < 3:
        return False
    if any(w in t for w in TITLE_NOISE):
        return False
    # must contain at least one allow keyword
    return any(k in t for k in TITLE_ALLOW_KEYWORDS)

def normalize_skill_token(tok: str) -> str:
    s = tok.strip().lower()
    if not s:
        return ""
    # drop obvious noise/non-tech tokens
    if s in SKILL_NOISE:
        return ""
    # keep alnum and common tech symbols
    import re
    s = re.sub(r"[^a-z0-9+#\.\-]", "", s)
    if len(s) < 2:
        return ""
    return s
@st.cache_data(show_spinner=False)
def load_job_postings() -> pd.DataFrame:
    path = Path("data/bronze/kaggle/job_postings")
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame()

def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

if section == "Skill Analysis":
    st.title("📈 Market Trends: Jobs and Skills")
    
    with st.spinner("Loading job market data..."):
        df_posts = load_job_postings()
    if df_posts.empty:
        st.warning("No job postings found at data/bronze/kaggle/job_postings. Run ETL first.")
    else:
        # Try to use a date column to focus on recent period
        date_col = pick_first_existing(
            df_posts,
            [
                "posted_date",
                "date_posted",
                "posting_date",
                "created_at",
                "created",
                "post_date",
            ],
        )
        if date_col:
            # Coerce to datetime and filter to recent N days
            df_posts[date_col] = pd.to_datetime(df_posts[date_col], errors="coerce")
            recent_days = st.slider("Lookback (days)", 7, 90, 30)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
            recent_df = df_posts[df_posts[date_col] >= cutoff].copy()
            base_df = recent_df if len(recent_df) > 100 else df_posts
            st.caption(
                f"Using {'recent' if len(recent_df)>100 else 'all'} data (rows={len(base_df):,})"
            )
        else:
            base_df = df_posts
            st.caption(f"No date column found; using all data (rows={len(base_df):,})")

        # Title column candidates
        title_col = pick_first_existing(
            base_df,
            ["title", "job_title", "position", "role", "posting_title"],
        )
        # Skills column candidates
        skills_col = pick_first_existing(
            base_df,
            ["skills", "key_skills", "skills_mentioned", "tags", "tech_stack"],
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Trending Job Titles")
            if title_col:
                titles = (
                    base_df[title_col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .value_counts()
                    .head(20)
                    .reset_index()
                )
                titles.columns = ["job_title", "count"]
                fig_t = px.bar(titles, x="job_title", y="count", template="plotly_white")
                fig_t.update_layout(xaxis_title="Job Title", yaxis_title="Postings", xaxis_tickangle=-30)
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("No job title column found.")

        with c2:
            st.subheader("Most Demanded Skills")
            if skills_col:
                # Split skills on common delimiters
                ser = (
                    base_df[skills_col]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .str.replace("/", ",", regex=False)
                    .str.replace(";", ",", regex=False)
                )
                exploded = ser.str.split(",").explode().str.strip()
                exploded = exploded[exploded.str.len() > 0]
                top_sk = exploded.value_counts().head(25).reset_index()
                top_sk.columns = ["skill", "count"]
                fig_s = px.bar(top_sk, x="skill", y="count", template="plotly_white")
                fig_s.update_layout(xaxis_title="Skill", yaxis_title="Mentions", xaxis_tickangle=-30)
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.info("No skills column found.")

        st.markdown("---")
        st.subheader("Recent Posting Volume")
        if date_col:
            vol = (
                base_df[[date_col]]
                .dropna()
                .assign(day=lambda d: d[date_col].dt.date)
                .groupby("day")
                .size()
                .reset_index(name="postings")
            )
            fig_v = px.line(vol, x="day", y="postings", template="plotly_white")
            st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.caption("Add a date column to see time trends.")

    st.markdown("---")
    st.subheader("StackOverflow: Top Languages by Year")
    gold_so_langs = get_gold_data("so_languages")
    if not gold_so_langs.empty and {"year","language","count"}.issubset(gold_so_langs.columns):
        year_sel = st.selectbox("Year", sorted(gold_so_langs["year"].dropna().unique()))
        subset = gold_so_langs[gold_so_langs["year"] == year_sel].head(30)
        fig_so = px.bar(subset, x="language", y="count", template="plotly_white")
        fig_so.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_so, use_container_width=True)
    else:
        st.caption("Run Spark ETL for StackOverflow aggregates.")

    st.markdown("---")
    st.subheader("Skill Demand Forecast (6–12 months)")
    # Suggest a default skill from skills_demand if available
    # Get default skill from skills demand data
    skills_data = get_gold_data("skills_demand")
    default_skill = skills_data.head(1)["skill"].iloc[0] if not skills_data.empty and "skill" in skills_data.columns else "python"
    skill = st.text_input("Skill", value=default_skill)
    horizon = st.slider("Horizon (months)", 6, 12, 12)
    if st.button("Get Forecast", use_container_width=True):
        fc = call_api_forecast(skill, horizon)
        pts = fc.get("forecast", [])
        if len(pts) > 0:
            df_fc = pd.DataFrame(pts)
            fig_fc = px.line(df_fc, x="date", y="forecast", template="plotly_white")
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("No forecast available for that skill yet.")

# Market Overview
# ------------------------------------------------------------
if section == "Market Overview":
    st.title("📊 Market Overview")
    posts = load_unified_postings()

    # High-level KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Job Postings", f"{len(posts):,}" if not posts.empty else "—")
    with c2:
        if not posts.empty and "location" in posts:
            kpi_card("Locations", f"{posts['location'].nunique():,}")
        else:
            kpi_card("Locations", "—")
    with c3:
        if not posts.empty and "remote_flag" in posts:
            pct = (posts["remote_flag"].mean() * 100.0) if len(posts) else 0
            kpi_card("Remote Share", f"{pct:.1f}%")
        else:
            kpi_card("Remote Share", "—")
    with c4:
        if not posts.empty and "salary_amount" in posts:
            avg = pd.to_numeric(posts["salary_amount"], errors="coerce").mean()
            kpi_card("Avg Salary", f"${avg:,.0f}" if pd.notna(avg) else "—")
        else:
            kpi_card("Avg Salary", "—")

    st.markdown("---")
    # Salary distribution + pay period share
    a, b = st.columns([2, 1])
    with a:
        st.subheader("Salary Distribution")
        if not posts.empty and "salary_amount" in posts:
            s = pd.to_numeric(posts["salary_amount"], errors="coerce")
            fig = px.histogram(posts.assign(salary=s), x="salary", nbins=60, template="plotly_white")
            # quantile markers
            q25, q50, q75 = s.quantile([0.25, 0.5, 0.75])
            fig.add_vline(x=q25, line_dash="dot", line_color="#94a3b8")
            fig.add_vline(x=q50, line_dash="dash", line_color="#0ea5e9")
            fig.add_vline(x=q75, line_dash="dot", line_color="#94a3b8")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No salary data available.")
    with b:
        st.subheader("Pay Period")
        if not posts.empty and "pay_period" in posts:
            share = posts["pay_period"].astype(str).value_counts().reset_index()
            share.columns = ["period", "count"]
            figp = px.pie(share, names="period", values="count", hole=0.5, template="plotly_white")
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.caption("No pay period data.")

    st.markdown("---")
    # Top titles and locations
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Top Job Titles")
        if not posts.empty and "job_title" in posts:
            top_titles = posts["job_title"].astype(str).str.title().value_counts().head(15)
            figt = px.bar(x=top_titles.values, y=top_titles.index, orientation="h", template="plotly_white")
            figt.update_layout(xaxis_title="Postings", yaxis_title="Job Title", height=450)
            st.plotly_chart(figt, use_container_width=True)
        else:
            st.caption("No titles available.")
    with t2:
        st.subheader("Top Locations")
        if not posts.empty and "location" in posts:
            loc = posts["location"].astype(str).str.strip()
            top_loc = loc.value_counts().head(15)
            figl = px.bar(x=top_loc.values, y=top_loc.index, orientation="h", template="plotly_white")
            figl.update_layout(xaxis_title="Postings", yaxis_title="Location", height=450)
            st.plotly_chart(figl, use_container_width=True)
        else:
            st.caption("No locations available.")

    st.markdown("---")
    # Monthly postings trend
    st.subheader("Hiring Trend (Monthly)")
    if not posts.empty and "posted_date" in posts:
        dt = pd.to_datetime(posts["posted_date"], errors="coerce")
        trend = (
            pd.DataFrame({"month": dt.dt.to_period("M").astype(str)})
            .dropna()
            .groupby("month")
            .size()
            .reset_index(name="postings")
        )
        figm = px.line(trend, x="month", y="postings", template="plotly_white")
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.caption("No posting dates available.")

    # In-demand skills snapshot
    # Optional sections: only show when available to avoid noise
    # Load skills data
    skills_data = get_gold_data("skills_demand")
    if not skills_data.empty and {"skill","count"}.issubset(skills_data.columns):
        st.markdown("---")
        st.subheader("Most In-demand Skills (Gold)")
        fig = px.bar(skills_data.head(25), x="skill", y="count", template="plotly_white")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    # Load BLS health data
    bls_data = get_gold_data("bls_health_index")
    if not bls_data.empty and {"year","health_index"}.issubset(bls_data.columns):
        st.markdown("---")
        st.subheader("Industry Growth (BLS Health Index)")
        fig_bls = px.line(bls_data.sort_values("year"), x="year", y="health_index", template="plotly_white")
        st.plotly_chart(fig_bls, use_container_width=True)

# ------------------------------------------------------------
# Salary Intelligence
# ------------------------------------------------------------
elif section == "Salary Intelligence":
    st.title("💰 Salary Intelligence")
    df_salaries = load_unified_salaries()
    if df_salaries.empty or "salary_amount" not in df_salaries:
        st.warning("No salary data available. Run the ETL first.")
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
            cur = df_salaries["currency"].value_counts().reset_index()
            cur.columns = ["currency", "count"]
            fig2 = px.bar(cur.head(10), x="currency", y="count", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Top Salaries (sample)")
        topn = st.slider("Rows", 5, 50, 10)
        st.dataframe(
            df_salaries.sort_values("salary_amount", ascending=False).head(topn),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Salary Calculator")
    years = st.slider("Years of Experience", 0, 30, 5)
    currency = st.selectbox("Currency", sorted(df_salaries["currency"].dropna().unique()) if not df_salaries.empty and "currency" in df_salaries else ["USD"])  # noqa: E501
    period = st.selectbox("Pay Period", sorted(df_salaries["period"].dropna().unique()) if not df_salaries.empty and "period" in df_salaries else ["YEARLY"])  # noqa: E501
    if st.button("Predict Salary", use_container_width=True):
        try:
            result = call_api_predict(years, currency, period)
            st.success(f"Predicted Salary: ${result.get('predicted_salary', '—'):,.0f}")
            st.json(result)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    elif section == "Career Planner":
        st.title("🎯 Smart Career Planner")
        st.markdown("Get **personalized** skill recommendations based on your current skills and target role.")
        
        col1, col2 = st.columns(2)
        with col1:
            current = st.text_input("Your current skills (comma-separated)", value="swift, ios frameworks", help="Enter skills separated by commas")
        with col2:
            target = st.text_input("Target role (optional)", value="iOS Engineer", help="What role are you targeting?")
        
        if st.button("Get Smart Recommendations", use_container_width=True, type="primary"):
            if not current.strip():
                st.warning("Please enter your current skills")
            else:
                with st.spinner("🤖 Analyzing your skills and generating personalized recommendations..."):
                    skills = [s.strip().lower() for s in current.split(",") if len(s.strip()) > 0]
                    recs = call_api_career_recs(skills, target)
                    
                    if recs.get("error"):
                        st.error(f"Error: {recs['error']}")
                    else:
                        # Display personalized results
                        st.success(f"🎉 Found {len(recs.get('recommended_skills', []))} personalized recommendations!")
                        
                        # Show categorized recommendations
                        categorized = recs.get("categorized_recommendations", {})
                        learning_path = recs.get("learning_path", {})
                        
                        if categorized:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.subheader("🔥 Must Learn")
                                st.markdown("*Critical for your target role*")
                                must_learn = categorized.get("must_learn", [])
                                if must_learn:
                                    for skill in must_learn[:5]:
                                        st.write(f"• **{skill.title()}**")
                                else:
                                    st.info("No must-have skills identified")
                            
                            with col2:
                                st.subheader("📈 Should Learn")
                                st.markdown("*Important for career growth*")
                                should_learn = categorized.get("should_learn", [])
                                if should_learn:
                                    for skill in should_learn[:5]:
                                        st.write(f"• **{skill.title()}**")
                                else:
                                    st.info("No should-have skills identified")
                            
                            with col3:
                                st.subheader("✨ Nice to Have")
                                st.markdown("*Future career opportunities*")
                                nice_to_have = categorized.get("nice_to_have", [])
                                if nice_to_have:
                                    for skill in nice_to_have[:5]:
                                        st.write(f"• **{skill.title()}**")
                                else:
                                    st.info("No nice-to-have skills identified")
                        
                        # Learning path section
                        if learning_path:
                            st.markdown("---")
                            st.subheader("🎯 Your Learning Path")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Phase 1: Immediate Focus**")
                                immediate = learning_path.get("immediate_focus", [])
                                for skill in immediate:
                                    st.write(f"• {skill.title()}")
                            
                            with col2:
                                st.markdown("**Phase 2: Next Phase**")
                                next_phase = learning_path.get("next_phase", [])
                                for skill in next_phase:
                                    st.write(f"• {skill.title()}")
                            
                            with col3:
                                st.markdown("**Phase 3: Future Goals**")
                                future = learning_path.get("future_goals", [])
                                for skill in future:
                                    st.write(f"• {skill.title()}")
                        
                        # Show analysis summary
                        st.markdown("---")
                        st.subheader("📊 Analysis Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Skills Analyzed", f"{recs.get('total_skills_analyzed', 0):,}")
                        with col2:
                            st.metric("Your Current Skills", len(skills))
                        with col3:
                            st.metric("Recommendations", len(recs.get('recommended_skills', [])))
                        
                        # Show all recommendations in a nice format
                        all_recs = recs.get("recommended_skills", [])
                        if all_recs:
                            st.markdown("---")
                            st.subheader("📋 All Recommendations")
                            
                            # Create a nice grid display
                            cols = st.columns(4)
                            for i, skill in enumerate(all_recs[:20]):  # Show top 20
                                with cols[i % 4]:
                                    st.write(f"**{i+1}.** {skill.title()}")
                            
                            if len(all_recs) > 20:
                                st.caption(f"... and {len(all_recs) - 20} more skills to explore")
                        else:
                            st.info("No recommendations available. Please check your input and try again.")

# ------------------------------------------------------------
# Real-time Trends
# ------------------------------------------------------------
elif section == "Real-time Trends":
    st.title("⚡ Real-time Trends")
    st.caption("Live job market trends and emerging opportunities")
    
    # Load job postings data for trending analysis
    @st.cache_data(show_spinner=False)
    def load_job_postings():
        try:
            import glob
            files = glob.glob("data/bronze/job_postings/*.parquet")
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
                return df
        except Exception:
            pass
        return pd.DataFrame()
    
    job_postings = load_job_postings()
    
    # Top row: Trending Job Titles and Skills
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🔥 Trending Job Titles")
        if not job_postings.empty and "title" in job_postings.columns:
            # Clean, filter, and count job titles
            titles_raw = job_postings["title"].dropna().astype(str)
            titles = titles_raw[titles_raw.str.len() > 0]
            titles = titles[titles.apply(is_real_job_title)]
            title_counts = titles.str.title().value_counts().head(15)

            if not title_counts.empty:
                fig_titles = px.bar(
                    x=title_counts.values,
                    y=title_counts.index,
                    orientation='h',
                    template="plotly_white",
                    title="Most Posted Job Titles"
                )
                fig_titles.update_layout(
                    xaxis_title="Number of Postings",
                    yaxis_title="Job Title",
                    height=400
                )
                st.plotly_chart(fig_titles, use_container_width=True)

                # Show top 5 as cards
                st.markdown("**Top 5 Trending Jobs:**")
                for i, (title, count) in enumerate(title_counts.head(5).items(), 1):
                    st.markdown(f"{i}. **{title}** - {count} postings")
            else:
                st.info("No job titles found in the data.")
        else:
            st.info("Job postings data not available.")
    
    with c2:
        st.subheader("💻 Trending Skills")
        if not job_postings.empty and "skills_desc" in job_postings.columns:
            # Extract skills from skills_desc column
            skills_text = job_postings["skills_desc"].dropna().astype(str)
            if not skills_text.empty:
                # Split on common delimiters and clean
                all_skills: list[str] = []
                for text in skills_text:
                    tokens = (
                        str(text)
                        .lower()
                        .replace(",", " ")
                        .replace(";", " ")
                        .replace("/", " ")
                        .split()
                    )
                    all_skills.extend([normalize_skill_token(s) for s in tokens])
                all_skills = [s for s in all_skills if s]

                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(15)
                    fig_skills = px.bar(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        orientation='h',
                        template="plotly_white",
                        title="Most Mentioned Skills"
                    )
                    fig_skills.update_layout(
                        xaxis_title="Mentions",
                        yaxis_title="Skill",
                        height=400
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)
                else:
                    st.info("No skills extracted from job descriptions.")
            else:
                st.info("No skills data available.")
        else:
            st.info("Skills data not available.")
    
    st.markdown("---")
    
    # Second row: Location and Company trends
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("📍 Hot Job Locations")
        if not job_postings.empty and "location" in job_postings.columns:
            locations = job_postings["location"].dropna().str.strip()
            locations = locations[locations.str.len() > 0]
            location_counts = locations.value_counts().head(10)
            
            if not location_counts.empty:
                fig_locations = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    template="plotly_white",
                    title="Top Job Locations"
                )
                fig_locations.update_layout(
                    xaxis_title="Location",
                    yaxis_title="Number of Jobs",
                    xaxis_tickangle=-45,
                    height=350
                )
                st.plotly_chart(fig_locations, use_container_width=True)
            else:
                st.info("No location data available.")
        else:
            st.info("Location data not available.")
    
    with c4:
        st.subheader("🏢 Top Hiring Companies")
        if not job_postings.empty and "company_name" in job_postings.columns:
            companies = job_postings["company_name"].dropna().str.strip()
            companies = companies[companies.str.len() > 0]
            company_counts = companies.value_counts().head(10)
            
            if not company_counts.empty:
                fig_companies = px.bar(
                    x=company_counts.index,
                    y=company_counts.values,
                    template="plotly_white",
                    title="Companies with Most Job Postings"
                )
                fig_companies.update_layout(
                    xaxis_title="Company",
                    yaxis_title="Number of Jobs",
                    xaxis_tickangle=-45,
                    height=350
                )
                st.plotly_chart(fig_companies, use_container_width=True)
            else:
                st.info("No company data available.")
        else:
            st.info("Company data not available.")
    
    st.markdown("---")
    
    # Third row: GitHub trends and tech linkage
    c5, c6 = st.columns(2)
    
    with c5:
        st.subheader("📊 GitHub Activity Trends")
        if not gold_gh_hourly.empty and {"hour","event_type","count"}.issubset(gold_gh_hourly.columns):
            fig = px.line(gold_gh_hourly, x="hour", y="count", color="event_type", template="plotly_white")
            fig.update_layout(title="GitHub Events by Hour")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No GitHub hourly trends available.")
    
    with c6:
        st.subheader("🔗 Tech ↔ Jobs Correlation")
        if not gold_tech_job_linkage.empty and {"year_month","language","events","job_mentions"}.issubset(gold_tech_job_linkage.columns):
            lang2 = st.selectbox("Select Technology", sorted(gold_tech_job_linkage["language"].dropna().unique()))
            link = gold_tech_job_linkage[gold_tech_job_linkage["language"] == lang2]
            fig_lk = px.line(link, x="year_month", y=["events","job_mentions"], template="plotly_white")
            fig_lk.update_layout(
                title=f"{lang2} Activity vs Job Mentions",
                xaxis_title="Month",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_lk, use_container_width=True)
        else:
            st.caption("Tech-job linkage not available.")
    
    # Market alerts section
    st.markdown("---")
    st.subheader("🚨 Market Alerts")
    
    if not job_postings.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_jobs = len(job_postings)
            st.metric("Total Job Postings", f"{total_jobs:,}")
        
        with col2:
            if "remote_allowed" in job_postings.columns:
                # Handle different data types for remote_allowed
                remote_col = job_postings["remote_allowed"]
                if remote_col.dtype == bool:
                    remote_jobs = remote_col.sum()
                else:
                    # Convert to boolean if it's string/other type
                    remote_jobs = (remote_col.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])).sum()
                remote_pct = (remote_jobs / total_jobs * 100) if total_jobs > 0 else 0
                st.metric("Remote Jobs", f"{remote_pct:.1f}%")
        
        with col3:
            if "max_salary" in job_postings.columns:
                # Convert salary to numeric, handling mixed types
                salary_series = pd.to_numeric(job_postings["max_salary"], errors='coerce')
                avg_salary = salary_series.mean()
                st.metric("Avg Max Salary", f"${avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A")
    else:
        st.info("Job market data not available for alerts.")

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
else:
    st.title("ℹ️ About")
    st.markdown(
        """
        This dashboard showcases the end-to-end Big Data pipeline:
        - Ingestion (GitHub, StackOverflow, BLS, Kaggle)
        - Processing (Spark to Bronze/Silver)
        - Modeling (XGBoost + MLflow)
        - Serving (FastAPI + Streamlit)
        """
    )
