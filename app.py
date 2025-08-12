import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import duckdb
import requests
import json
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os

st.set_page_config(page_title="Talk to ESG Data")

# Configuration - Using Groq Cloud API
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""

# Sidebar - About Section
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### Sustainability Gen AI Application
    
    This AI-powered application helps analyze Environmental, Social, and Governance (ESG) data with focus on greenhouse gas emissions tracking.
    
    **Key Features:**
    - 🤖 Natural language queries to ESG data
    - 📊 Interactive visualizations
    - 🔍 Scope 1, 2, and 3 emissions analysis
    - 📈 Multi-year trend analysis
    - ⚡ Powered by Groq AI for intelligent SQL generation
    
    **Data Coverage:**
    - Reporting years: 2019-2023
    - Quarters: Q1, Q2, Q3, Q4
    - Emission scopes: 1, 2, and 3
    - Units: tCO2e (tonnes CO2 equivalent)
    - Real-time query processing
    
    Built with Streamlit, DuckDB, and LangChain for seamless ESG data insights.
    """)

st.title("📊 AI Powered GHG KPI Chatbot (MVP)")

# Initialize DuckDB connection
@st.cache_resource
def init_duckdb():
    conn = duckdb.connect(':memory:')
    return conn

# Load CSV data
@st.cache_data
def load_data():
    # Try to load from CSV file, fallback to sample data
    try:
        df = pd.read_csv('sample_emissions.csv')
        st.success("✅ Loaded data from sample_emissions.csv")
    except FileNotFoundError:
        # Fallback sample data with quarters
        years = [2019, 2020, 2021, 2022, 2023]
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        scopes = ['Scope 1', 'Scope 2', 'Scope 3']
        
        sample_data = {
            'reporting_year': years * 12,  # 5 years * 4 quarters * 3 scopes
            'quarter': quarters * 15,      # 4 quarters * 5 years * 3 scopes
            'disc_name': [scope for scope in scopes for _ in range(20)],  # Each scope repeated 20 times
            'emission': [120, 118, 125, 122, 110, 108, 115, 112, 95, 92, 98, 96, 85, 82, 88, 86, 75, 72, 78, 76,  # Scope 1
                        250, 245, 260, 255, 230, 225, 240, 235, 200, 195, 210, 205, 180, 175, 190, 185, 160, 155, 170, 165,  # Scope 2
                        450, 440, 470, 460, 420, 410, 440, 430, 380, 370, 400, 390, 340, 330, 360, 350, 300, 290, 320, 310],  # Scope 3
            'unit': ['tCO2e'] * 60
        }
        df = pd.DataFrame(sample_data)
        st.info("ℹ️ Using sample data (sample_emissions.csv not found)")
    
    return df

# Setup DuckDB with data
def setup_duckdb():
    conn = init_duckdb()
    df = load_data()
    conn.register('emissions', df)
    return conn, df

# LangChain prompt template for text-to-SQL
def create_sql_prompt():
    template = """Convert this question to SQL query for table 'emissions' with columns: reporting_year (INTEGER), quarter (VARCHAR), disc_name (VARCHAR), emission (FLOAT), unit (VARCHAR).

Question: {question}

SQL:"""
    return PromptTemplate(template=template, input_variables=["question"])

# Groq Cloud API call for text-to-SQL
def query_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a SQL expert. Convert natural language questions to SQL queries. Return only the SQL query, no explanations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            sql_query = result["choices"][0]["message"]["content"].strip()
            return sql_query
        return ""
    except Exception as e:
        return ""

# Enhanced rule-based SQL generation
def generate_sql_fallback(prompt):
    prompt_lower = prompt.lower()
    
    # Quarter-specific queries
    for quarter in ['q1', 'q2', 'q3', 'q4']:
        if quarter in prompt_lower:
            return f"SELECT * FROM emissions WHERE quarter = '{quarter.upper()}' ORDER BY reporting_year, disc_name"
    
    # Year-specific queries
    for year in [2019, 2020, 2021, 2022, 2023]:
        if str(year) in prompt_lower:
            return f"SELECT * FROM emissions WHERE reporting_year = {year} ORDER BY quarter, disc_name"
    
    # Scope-specific queries
    if "scope 1" in prompt_lower:
        return "SELECT * FROM emissions WHERE disc_name = 'Scope 1' ORDER BY reporting_year, quarter"
    elif "scope 2" in prompt_lower:
        return "SELECT * FROM emissions WHERE disc_name = 'Scope 2' ORDER BY reporting_year, quarter"
    elif "scope 3" in prompt_lower:
        return "SELECT * FROM emissions WHERE disc_name = 'Scope 3' ORDER BY reporting_year, quarter"
    
    # Range queries
    elif "2019 to 2023" in prompt_lower or "all years" in prompt_lower:
        return "SELECT * FROM emissions WHERE reporting_year BETWEEN 2019 AND 2023 ORDER BY reporting_year, quarter, disc_name"
    

    
    # Latest/recent queries
    elif "latest" in prompt_lower or "recent" in prompt_lower:
        return "SELECT * FROM emissions WHERE reporting_year = 2023 ORDER BY quarter, disc_name"
    
    # All data queries
    elif "all data" in prompt_lower or "everything" in prompt_lower or "show all" in prompt_lower:
        return "SELECT * FROM emissions ORDER BY reporting_year, quarter, disc_name"
    
    # Default query
    else:
        return "SELECT * FROM emissions ORDER BY reporting_year, quarter, disc_name LIMIT 10"

# Execute SQL query using DuckDB
def execute_query(conn, sql_query):
    try:
        result = conn.execute(sql_query).fetchdf()
        return result
    except Exception as e:
        st.error(f"SQL execution error: {str(e)}")
        return pd.DataFrame()

# Main app logic
def main():
    # Load data
    conn, sample_df = setup_duckdb()
    
    # Show sample data info
    with st.expander("📋 Available Data Preview"):
        st.write("Sample of available data:")
        st.dataframe(sample_df.head(10))
        st.write(f"Total records: {len(sample_df)}")
    
    # User input
    prompt = st.text_input("Ask your question (e.g., 'Show Q1 emissions' or 'What are Scope 1 emissions in 2023?'):")
    
    if st.button("Submit") and prompt:
        with st.spinner("Processing your query..."):
            
            # Create LangChain prompt
            sql_prompt_template = create_sql_prompt()
            formatted_prompt = sql_prompt_template.format(question=prompt)
            
            # Try Groq API first, fallback to rule-based patterns
            sql_query = ""
            if GROQ_API_KEY:
                sql_query = query_groq(formatted_prompt)
                if sql_query and "SELECT" in sql_query.upper():
                    st.success("✅ Generated SQL using Groq API")
            
            # Fallback to rule-based patterns if API fails or no key
            if not sql_query or "SELECT" not in sql_query.upper():
                sql_query = generate_sql_fallback(prompt)
                st.info("🔄 Using pattern-based SQL generation (API unavailable)")
            
            # Execute query
            if sql_query:
                result_df = execute_query(conn, sql_query)
                
                if not result_df.empty:
                    # Store in session state
                    st.session_state.df = result_df
                    st.session_state.sql = sql_query
                else:
                    st.warning("No data returned from query.")
            else:
                st.error("Could not generate SQL query.")
    
    # Display results
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        sql = st.session_state.sql
        
        st.subheader("🧠 Generated SQL")
        st.code(sql, language="sql")
        
        st.subheader("📋 Query Results")
        st.dataframe(df)
        
        # Show available columns
        with st.expander("Available Columns"):
            st.write(f"Columns: {list(df.columns)}")
        
        # Visualization (same as original)
        if "reporting_year" in df.columns and "emission" in df.columns:
            st.subheader("📊 Visualize Emission")
            
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    unique_years = sorted(df['reporting_year'].unique())
                    if len(unique_years) > 1:
                        selected_year = st.selectbox("Select Year", unique_years, key="year_selector")
                        df_filtered = df[df['reporting_year'] == selected_year]
                    else:
                        df_filtered = df
                
                with col2:
                    chart_type = st.selectbox("Select chart type", ["Bar", "Line", "Pie", "Donut"], key="chart_selector")
            
            with st.container():
                try:
                    if chart_type == "Bar":
                        if "disc_name" in df_filtered.columns:
                            chart = alt.Chart(df_filtered).mark_bar().encode(
                                x="disc_name:O",
                                y="emission:Q",
                                color="disc_name:N",
                                tooltip=list(df_filtered.columns)
                            ).properties(width=600, height=400)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Column 'disc_name' not found for bar chart.")
                    
                    elif chart_type == "Line":
                        if "disc_name" in df.columns:
                            chart = alt.Chart(df).mark_line().encode(
                                x="reporting_year:O",
                                y="emission:Q",
                                color="disc_name:N",
                                tooltip=list(df.columns)
                            ).properties(width=600, height=400)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Column 'disc_name' not found for line chart.")
                    
                    elif chart_type in ["Pie", "Donut"]:
                        if "disc_name" in df_filtered.columns:
                            fig = px.pie(
                                df_filtered,
                                names="disc_name",
                                values="emission",
                                hole=0.4 if chart_type == "Donut" else 0
                            )
                            fig.update_layout(width=600, height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Column 'disc_name' not found for pie/donut chart.")
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")

if __name__ == "__main__":
    main()