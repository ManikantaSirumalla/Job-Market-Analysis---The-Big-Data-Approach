import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="Smart Career Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cache API calls for better performance
@st.cache_data(ttl=300)  # Cache for 5 minutes
def call_career_api(skills, target_role):
    try:
        response = requests.post(
            "http://127.0.0.1:8001/career-recommendations",
            json={"current_skills": skills, "target_role": target_role},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Real data loading functions
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_skills_demand():
    try:
        df = pd.read_parquet('data/gold/skills_demand.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading skills demand: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_unified_salaries():
    try:
        if Path('data/silver/unified_salaries.parquet').exists():
            df = pd.read_parquet('data/silver/unified_salaries.parquet')
        else:
            df = pd.read_parquet('data/silver/unified_salaries')
        return df
    except Exception as e:
        st.error(f"Error loading salaries: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_job_postings():
    try:
        # Load a sample of job postings
        files = list(Path('data/bronze/job_postings').glob('*.parquet'))[:3]  # Load first 3 files
        if files:
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading job postings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_market_summary():
    try:
        salaries_df = load_unified_salaries()
        skills_df = load_skills_demand()
        jobs_df = load_job_postings()
        
        summary = {
            "total_jobs": len(jobs_df) if not jobs_df.empty else 0,
            "avg_salary": salaries_df['salary_amount'].mean() if not salaries_df.empty else 0,
            "top_skills": skills_df.head(5)['skill'].tolist() if not skills_df.empty else [],
            "trending_roles": jobs_df['title'].value_counts().head(5).index.tolist() if not jobs_df.empty else []
        }
        return summary
    except Exception as e:
        st.error(f"Error loading market summary: {e}")
        return {
            "total_jobs": 0,
            "avg_salary": 0,
            "top_skills": [],
            "trending_roles": []
        }

# Main app
def main():
    st.title("🎯 Smart Career Planner")
    st.markdown("Get personalized skill recommendations based on your current skills and target role.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Career Planner", 
        "Market Overview", 
        "Skill Analysis",
        "Salary Intelligence", 
        "Real-time Trends",
        "About"
    ])
    
    if page == "Career Planner":
        show_career_planner()
    elif page == "Market Overview":
        show_market_overview()
    elif page == "Skill Analysis":
        show_skill_analysis()
    elif page == "Salary Intelligence":
        show_salary_intelligence()
    elif page == "Real-time Trends":
        show_realtime_trends()
    else:
        show_about()

def show_career_planner():
    st.header("🎯 Personalized Career Recommendations")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        current_skills = st.text_input(
            "Your current skills (comma-separated)",
            value="swift, ios frameworks",
            help="Enter your current skills separated by commas"
        )
    
    with col2:
        target_role = st.text_input(
            "Target role (optional)",
            value="iOS Engineer",
            help="What role are you targeting?"
        )
    
    # Get recommendations button
    if st.button("🚀 Get Smart Recommendations", type="primary", use_container_width=True):
        if not current_skills.strip():
            st.warning("Please enter your current skills")
            return
        
        with st.spinner("🤖 Analyzing your skills and generating personalized recommendations..."):
            skills_list = [s.strip().lower() for s in current_skills.split(",") if s.strip()]
            result = call_career_api(skills_list, target_role)
            
            if result.get("error"):
                st.error(f"Error: {result['error']}")
                return
            
            # Display results
            display_recommendations(result)

def display_recommendations(data):
    st.success(f"🎉 Found {len(data.get('recommended_skills', []))} personalized recommendations!")
    
    # Categorized recommendations
    categorized = data.get("categorized_recommendations", {})
    
    if categorized:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔥 Must Learn")
            st.markdown("*Critical for your target role*")
            must_learn = categorized.get("must_learn", [])
            if must_learn:
                for skill in must_learn:
                    st.write(f"• **{skill.title()}**")
            else:
                st.info("No must-have skills identified")
        
        with col2:
            st.subheader("📈 Should Learn")
            st.markdown("*Important for career growth*")
            should_learn = categorized.get("should_learn", [])
            if should_learn:
                for skill in should_learn[:5]:  # Show top 5
                    st.write(f"• **{skill.title()}**")
            else:
                st.info("No should-have skills identified")
        
        with col3:
            st.subheader("✨ Nice to Have")
            st.markdown("*Future career opportunities*")
            nice_to_have = categorized.get("nice_to_have", [])
            if nice_to_have:
                for skill in nice_to_have[:5]:  # Show top 5
                    st.write(f"• **{skill.title()}**")
            else:
                st.info("No nice-to-have skills identified")
    
    # Learning path
    learning_path = data.get("learning_path", {})
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
    
    # Summary metrics
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Skills Analyzed", f"{data.get('total_skills_analyzed', 0):,}")
    with col2:
        st.metric("Your Current Skills", len(data.get('your_skills', [])))
    with col3:
        st.metric("Recommendations", len(data.get('recommended_skills', [])))

def show_market_overview():
    st.header("📊 Job Market Overview")
    
    # Load real data
    with st.spinner("Loading real market data..."):
        data = load_market_summary()
        skills_df = load_skills_demand()
        salaries_df = load_unified_salaries()
        jobs_df = load_job_postings()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", f"{data['total_jobs']:,}")
    with col2:
        st.metric("Average Salary", f"${data['avg_salary']:,.0f}")
    with col3:
        st.metric("Top Skills", len(data['top_skills']))
    with col4:
        st.metric("Trending Roles", len(data['trending_roles']))
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Skills (Real Data)")
        if not skills_df.empty:
            top_skills = skills_df.head(10)
            fig = px.bar(top_skills, x='skill', y='count', title="Most In-Demand Skills", 
                        color='count', color_continuous_scale='viridis')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skills data available")
    
    with col2:
        st.subheader("Trending Roles (Real Data)")
        if not jobs_df.empty:
            top_roles = jobs_df['title'].value_counts().head(10).reset_index()
            top_roles.columns = ['Role', 'Count']
            fig = px.bar(top_roles, x='Role', y='Count', title="Most Posted Job Titles",
                        color='Count', color_continuous_scale='plasma')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No job postings data available")
    
    # Additional real data insights
    if not salaries_df.empty:
        st.subheader("Salary Distribution (Real Data)")
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary by currency
            currency_counts = salaries_df['currency'].value_counts().head(5)
            fig = px.pie(values=currency_counts.values, names=currency_counts.index, 
                        title="Salary Distribution by Currency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary by period
            period_counts = salaries_df['period'].value_counts()
            fig = px.bar(x=period_counts.index, y=period_counts.values, 
                        title="Salary Distribution by Period")
            st.plotly_chart(fig, use_container_width=True)

def show_skill_analysis():
    st.header("📊 Skill Analysis")
    
    # Load real skills data
    with st.spinner("Loading real skills data..."):
        skills_df = load_skills_demand()
    
    if skills_df.empty:
        st.warning("No skills data available. Please run the ETL pipeline first.")
        return
    
    # Skill demand chart
    st.subheader("Top In-Demand Skills (Real Data)")
    top_skills = skills_df.head(15)
    fig = px.bar(top_skills, x='skill', y='count', title="Most In-Demand Skills", 
                color='count', color_continuous_scale='viridis')
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Skill analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Skills by Category (Real Data)")
        if 'category' in skills_df.columns:
            category_counts = skills_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, 
                        title="Skills Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category information not available")
    
    with col2:
        st.subheader("Top Skills Table")
        st.dataframe(
            skills_df[['skill', 'count', 'category']].head(10),
            use_container_width=True,
            hide_index=True
        )
    
    # Skill insights
    st.subheader("📈 Skill Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Skills", len(skills_df))
    with col2:
        st.metric("Total Demand", f"{skills_df['count'].sum():,}")
    with col3:
        st.metric("Avg Demand", f"{skills_df['count'].mean():.0f}")
    
    # Skill categories breakdown
    if 'category' in skills_df.columns:
        st.subheader("Skills by Category")
        category_skills = skills_df.groupby('category').agg({
            'skill': 'count',
            'count': 'sum'
        }).reset_index()
        category_skills.columns = ['Category', 'Skill Count', 'Total Demand']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(category_skills, x='Category', y='Skill Count', 
                        title="Number of Skills by Category")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(category_skills, x='Category', y='Total Demand', 
                        title="Total Demand by Category")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def show_salary_intelligence():
    st.header("💰 Salary Intelligence")
    
    # Load real salary data
    with st.spinner("Loading real salary data..."):
        salaries_df = load_unified_salaries()
        jobs_df = load_job_postings()
    
    if salaries_df.empty:
        st.warning("No salary data available. Please run the ETL pipeline first.")
        return
    
    # Salary calculator
    st.subheader("Salary Calculator (Based on Real Data)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years_exp = st.slider("Years of Experience", 0, 20, 5)
    with col2:
        available_currencies = salaries_df['currency'].unique().tolist() if not salaries_df.empty else ["USD"]
        currency = st.selectbox("Currency", available_currencies)
    with col3:
        available_periods = salaries_df['period'].unique().tolist() if not salaries_df.empty else ["YEARLY"]
        period = st.selectbox("Period", available_periods)
    
    if st.button("Calculate Salary", type="primary"):
        with st.spinner("Calculating salary based on real data..."):
            # Filter by currency and period
            filtered_salaries = salaries_df[
                (salaries_df['currency'] == currency) & 
                (salaries_df['period'] == period)
            ]
            
            if not filtered_salaries.empty:
                base_salary = filtered_salaries['salary_amount'].mean()
                # Simple experience adjustment (this could be more sophisticated)
                adjusted_salary = base_salary * (1 + (years_exp * 0.1))
                st.success(f"Estimated Salary: {currency} {adjusted_salary:,.0f} {period}")
                st.info(f"Based on {len(filtered_salaries)} real salary records")
            else:
                st.warning(f"No salary data found for {currency} {period}")
    
    # Real salary distributions
    st.subheader("Salary Distributions (Real Data)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by currency
        currency_stats = salaries_df.groupby('currency')['salary_amount'].agg(['mean', 'min', 'max']).reset_index()
        currency_stats.columns = ['Currency', 'Mean', 'Min', 'Max']
        
        fig = px.bar(currency_stats, x='Currency', y=['Min', 'Mean', 'Max'], 
                     title="Salary Ranges by Currency", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary by period
        period_stats = salaries_df.groupby('period')['salary_amount'].agg(['mean', 'min', 'max']).reset_index()
        period_stats.columns = ['Period', 'Mean', 'Min', 'Max']
        
        fig = px.bar(period_stats, x='Period', y=['Min', 'Mean', 'Max'], 
                     title="Salary Ranges by Period", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Salary insights
    st.subheader("📊 Salary Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(salaries_df):,}")
    with col2:
        st.metric("Average Salary", f"${salaries_df['salary_amount'].mean():,.0f}")
    with col3:
        st.metric("Median Salary", f"${salaries_df['salary_amount'].median():,.0f}")
    with col4:
        st.metric("Max Salary", f"${salaries_df['salary_amount'].max():,.0f}")
    
    # Salary distribution histogram
    st.subheader("Salary Distribution")
    fig = px.histogram(salaries_df, x='salary_amount', nbins=50, 
                       title="Salary Distribution (All Records)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top paying currencies and periods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Paying Currencies")
        top_currencies = salaries_df.groupby('currency')['salary_amount'].mean().sort_values(ascending=False).head(5)
        fig = px.bar(x=top_currencies.index, y=top_currencies.values, 
                     title="Average Salary by Currency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Paying Periods")
        top_periods = salaries_df.groupby('period')['salary_amount'].mean().sort_values(ascending=False)
        fig = px.bar(x=top_periods.index, y=top_periods.values, 
                     title="Average Salary by Period")
        st.plotly_chart(fig, use_container_width=True)

def show_realtime_trends():
    st.header("📈 Real-time Trends")
    
    # Load real data
    with st.spinner("Loading real market trends..."):
        jobs_df = load_job_postings()
        skills_df = load_skills_demand()
        salaries_df = load_unified_salaries()
    
    if jobs_df.empty:
        st.warning("No job postings data available. Please run the ETL pipeline first.")
        return
    
    # Trending jobs (real data)
    st.subheader("🔥 Trending Job Titles (Real Data)")
    
    # Get top job titles
    top_jobs = jobs_df['title'].value_counts().head(10).reset_index()
    top_jobs.columns = ['Job Title', 'Postings']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(top_jobs, x='Job Title', y='Postings', title="Most Posted Job Titles",
                    color='Postings', color_continuous_scale='viridis')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show job titles table
        st.subheader("Top Job Titles")
        st.dataframe(top_jobs, use_container_width=True, hide_index=True)
    
    # Trending skills (real data)
    st.subheader("🚀 Trending Skills (Real Data)")
    
    if not skills_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills demand scatter
            fig = px.scatter(skills_df.head(15), x='skill', y='count', size='count', 
                            hover_name='skill', title="Skill Demand Distribution",
                            color='count', color_continuous_scale='plasma')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Skills by category
            if 'category' in skills_df.columns:
                category_counts = skills_df['category'].value_counts()
                fig = px.pie(values=category_counts.values, names=category_counts.index, 
                            title="Skills by Category")
                st.plotly_chart(fig, use_container_width=True)
    
    # Market insights (real data)
    st.subheader("📊 Market Insights (Real Data)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Job Postings", f"{len(jobs_df):,}")
    with col2:
        avg_salary = salaries_df['salary_amount'].mean() if not salaries_df.empty else 0
        st.metric("Average Salary", f"${avg_salary:,.0f}")
    with col3:
        st.metric("Unique Companies", f"{jobs_df['company_name'].nunique():,}")
    with col4:
        st.metric("Skills Tracked", f"{len(skills_df):,}")
    
    # Company insights
    st.subheader("🏢 Company Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top companies by job postings
        top_companies = jobs_df['company_name'].value_counts().head(10).reset_index()
        top_companies.columns = ['Company', 'Job Postings']
        
        fig = px.bar(top_companies, x='Company', y='Job Postings', 
                     title="Top Companies by Job Postings",
                     color='Job Postings', color_continuous_scale='blues')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Job posting distribution
        if 'max_salary' in jobs_df.columns:
            # Filter out invalid salaries
            valid_salaries = jobs_df[jobs_df['max_salary'].notna() & (jobs_df['max_salary'] > 0)]
            if not valid_salaries.empty:
                fig = px.histogram(valid_salaries, x='max_salary', nbins=30, 
                                 title="Salary Distribution in Job Postings")
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent trends analysis
    st.subheader("📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Skills demand ranking
        if not skills_df.empty:
            st.subheader("Top 10 Skills by Demand")
            top_skills_display = skills_df.head(10)[['skill', 'count', 'category']]
            st.dataframe(top_skills_display, use_container_width=True, hide_index=True)
    
    with col2:
        # Job title word cloud (simplified)
        st.subheader("Most Common Job Title Words")
        if not jobs_df.empty:
            # Extract common words from job titles
            all_titles = ' '.join(jobs_df['title'].dropna().astype(str))
            words = all_titles.lower().split()
            # Filter out common words
            common_words = ['engineer', 'developer', 'analyst', 'manager', 'specialist', 'senior', 'junior']
            word_counts = {}
            for word in words:
                if len(word) > 3 and word in common_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            if word_counts:
                word_df = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])
                word_df = word_df.sort_values('Count', ascending=False)
                
                fig = px.bar(word_df, x='Word', y='Count', title="Most Common Job Title Words")
                st.plotly_chart(fig, use_container_width=True)

def show_about():
    st.header("About Smart Career Planner")
    
    st.markdown("""
    ## 🎯 What This Tool Does
    
    The Smart Career Planner provides **personalized skill recommendations** based on:
    - Your current skills
    - Your target role
    - Job market demand
    - Skill progression paths
    
    ## 🚀 Key Features
    
    - **Dynamic Recommendations**: Different suggestions for each role
    - **Categorized Learning**: Must Learn, Should Learn, Nice to Have
    - **Learning Paths**: Structured progression from immediate to future goals
    - **Real-time Analysis**: Based on current job market data
    
    ## 🎯 Supported Roles
    
    - **iOS Engineer**: Swift, UIKit, SwiftUI, Core Data, ARKit
    - **Data Scientist**: Python, TensorFlow, PyTorch, SQL, AWS
    - **Web Developer**: JavaScript, React, Node.js, HTML, CSS
    - **DevOps Engineer**: Docker, Kubernetes, AWS, Linux, CI/CD
    
    ## 🔧 Technical Stack
    
    - **Backend**: FastAPI with intelligent skill matching
    - **Frontend**: Streamlit with beautiful visualizations
    - **Data**: Real job market data and skill databases
    - **AI**: Smart priority scoring and role-specific recommendations
    """)
    
    st.info("💡 **Tip**: Enter your current skills and target role to get personalized recommendations!")

if __name__ == "__main__":
    main()
