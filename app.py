def basic_visualization_fallback(df):
    """Basic visualization when advanced module is not available"""
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap", "Pie Chart"]
    )
    
    if viz_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis:", df.columns)
        with col2:
            y_axis = st.selectbox("Select Y-axis:", df.columns)
        
        if st.button("Generate Chart"):
            if viz_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
            elif viz_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
            elif viz_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type in ["Histogram", "Box Plot"]:
        column = st.selectbox("Select Column:", df.select_dtypes(include=[np.number]).columns)
        
        if st.button("Generate Chart"):
            if viz_type == "Histogram":
                fig = px.histogram(df, x=column, title=f'Distribution of {column}')
            elif viz_type == "Box Plot":
                fig = px.box(df, y=column, title=f'Box Plot of {column}')
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            if st.button("Generate Heatmap"):
                corr_matrix = df[numerical_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numerical columns for heatmap.")
    
    elif viz_type == "Pie Chart":
        column = st.selectbox("Select Categorical Column:", df.select_dtypes(include=['object']).columns)
        
        if st.button("Generate Pie Chart"):
            value_counts = df[column].value_counts().head(10)  # Top 10 categories
            fig = px.pie(values=value_counts.values, names=value_counts.index, 
                        title=f'Distribution of {column}')
            st.plotly_chart(fig, use_container_width=True)

def natural_language_sql():
    st.markdown('<div class="section-header">💬 Enhanced Natural Language to SQL</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    
    # Try to import advanced NLP module
    try:
        from nlp_to_sql import NLPToSQL
        nlp_converter = NLPToSQL(df)
        use_advanced = True
    except ImportError:
        use_advanced = False
        st.info("Advanced NLP module not found. Using basic NLP to SQL conversion.")
    
    st.write("Ask questions about your data in plain English!")
    
    # Create two columns for examples and input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💡 Example Questions:")
        examples = [
            "Show me the top 10 records",
            "What is the average salary?",
            "Count unique customers",
            "Show records where age is greater than 25",
            "Group by department and count employees",
            "What is the maximum revenue?",
            "Show the bottom 5 performing regions",
            "Find records with missing email addresses"
        ]
        
        selected_example = st.selectbox("Choose an example:", ["Custom Question"] + examples)
        
        if selected_example != "Custom Question":
            st.code(selected_example)
    
    with col2:
        st.subheader("📊 Dataset Information:")
        st.write(f"**Columns:** {', '.join(df.columns.tolist()[:10])}")
        if len(df.columns) > 10:
            st.write(f"... and {len(df.columns) - 10} more columns")
        
        st.write(f"**Numeric Columns:** {', '.join(df.select_dtypes(include=[np.number]).columns.tolist()[:5])}")
        st.write(f"**Categorical Columns:** {', '.join(df.select_dtypes(include=['object']).columns.tolist()[:5])}")
    
    # Question input
    if selected_example == "Custom Question":
        user_question = st.text_input("🤔 Your Question:", placeholder="e.g., Show me records where age is greater than 25")
    else:
        user_question = st.text_input("🤔 Your Question:", value=selected_example)
    
    # Additional options
    with st.expander("🔧 Advanced Options"):
        show_sql = st.checkbox("Show generated SQL query", value=True)
        limit_results = st.number_input("Limit results to:", min_value=1, max_value=10000, value=100)
        export_results = st.checkbox("Enable result export")
    
    if st.button("🔍 Generate SQL & Execute"):
        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    if use_advanced:
                        # Use advanced NLP to SQL
                        sql_query = nlp_converter.generate_sql(user_question)
                        result = nlp_converter.execute_pandas_equivalent(user_question)
                    else:
                        # Use basic conversion
                        sql_query = generate_sql_from_question(user_question, df.columns.tolist())
                        result = execute_pandas_query(df, user_question)
                    
                    # Limit results
                    if result is not None and len(result) > limit_results:
                        result = result.head(limit_results)
                        st.info(f"Showing first {limit_results} results out of {len(result)} total matches")
                    
                    if show_sql and sql_query:
                        st.subheader("🔧 Generated SQL Query:")
                        st.code(sql_query, language='sql')
                    
                    st.subheader("📊 Query Results:")
                    if result is not None and not result.empty:
                        # Display results
                        st.dataframe(result, use_container_width=True)
                        
                        # Result statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Result Rows", len(result))
                        with col2:
                            st.metric("Result Columns", len(result.columns))
                        with col3:
                            if result.select_dtypes(include=[np.number]).columns.any():
                                numeric_col = result.select_dtypes(include=[np.number]).columns[0]
                                avg_val = result[numeric_col].mean()
                                st.metric(f"Avg {numeric_col}", f"{avg_val:.2f}")
                        
                        # Export options
                        if export_results:
                            st.write("**📁 Export Results:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                csv = result.to_csv(index=False)
                                st.download_button("📊 CSV", csv, "query_results.csv", "text/csv")
                            
                            with col2:
                                json_str = result.to_json(orient='records', indent=2)
                                st.download_button("📄 JSON", json_str, "query_results.json", "application/json")
                            
                            with col3:
                                excel_buffer = BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                    result.to_excel(writer, sheet_name='Query_Results', index=False)
                                excel_data = excel_buffer.getvalue()
                                st.download_button("📈 Excel", excel_data, "query_results.xlsx", 
                                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        
                        # Quick visualization if applicable
                        if len(result) > 1 and st.checkbox("📊 Quick Visualization"):
                            numeric_cols = result.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                viz_col = st.selectbox("Select column to visualize:", numeric_cols)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig_hist = px.histogram(result, x=viz_col, title=f"Distribution of {viz_col}")
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                with col2:
                                    fig_box = px.box(result, y=viz_col, title=f"Box Plot of {viz_col}")
                                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    else:
                        st.error("❌ No results found or query couldn't be executed")
                        st.info("💡 Try rephrasing your question or check if the column names are correct")
                        
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
                    st.info("💡 Tips for better results:")
                    st.info("- Use exact column names from your dataset")
                    st.info("- Be specific about what you want to see")
                    st.info("- Try simpler questions first")
        else:
            st.warning("⚠️ Please enter a question!")
    
    # Query History (if you want to add this feature)
    if st.checkbox("📚 Show Query Examples for Your Dataset"):
        st.subheader("🎯 Suggested Queries for Your Data:")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        suggestions = []
        
        if numeric_cols:
            suggestions.append(f"What is the average {numeric_cols[0]}?")
            suggestions.append(f"Show me the top 10 records by {numeric_cols[0]}")
            if len(numeric_cols) > 1:
                suggestions.append(f"Show records where {numeric_cols[0]} is greater than {df[numeric_cols[0]].mean():.0f}")
        
        if categorical_cols:
            suggestions.append(f"Count unique values in {categorical_cols[0]}")
            suggestions.append(f"Group by {categorical_cols[0]} and show counts")
            if numeric_cols:
                suggestions.append(f"What is the average {numeric_cols[0]} by {categorical_cols[0]}?")
        
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")
            if st.button(f"Try this query", key=f"suggestion_{i}"):
                st.session_state.suggested_query = suggestion
                st.experimental_rerun()

# Import BytesIO for Excel export
from io import BytesIO 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import sqlite3
import openai
from io import StringIO, BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚀 AI Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

def main():
    st.markdown('<h1 class="main-header">🚀 AI-Powered Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Your One-Stop Solution for Data Analysis, Cleaning, SQL Queries, Visualization & Machine Learning!")
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    options = [
        "🏠 Home",
        "📤 Data Upload",
        "🔍 EDA (Exploratory Data Analysis)",
        "🧹 Data Cleaning",
        "💬 Natural Language to SQL",
        "📊 Visualization Dashboard",
        "🤖 Machine Learning",
        "📚 Tutorial & Help"
    ]
    
    choice = st.sidebar.selectbox("Select Option:", options)
    
    # Add dataset info in sidebar if data is loaded
    if st.session_state.data is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Dataset")
        st.sidebar.write(f"**Rows:** {len(st.session_state.data)}")
        st.sidebar.write(f"**Columns:** {len(st.session_state.data.columns)}")
        
        if st.session_state.cleaned_data is not None:
            st.sidebar.success("✅ Cleaned Data Available")
        
        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        if st.sidebar.button("🔄 Reset All Data"):
            st.session_state.data = None
            st.session_state.cleaned_data = None
            st.experimental_rerun()
    
    if choice == "🏠 Home":
        show_home()
    elif choice == "📤 Data Upload":
        data_upload_section()
    elif choice == "🔍 EDA (Exploratory Data Analysis)":
        eda_section()
    elif choice == "🧹 Data Cleaning":
        data_cleaning_section()
    elif choice == "💬 Natural Language to SQL":
        natural_language_sql()
    elif choice == "📊 Visualization Dashboard":
        visualization_section()
    elif choice == "🤖 Machine Learning":
        machine_learning_section()
   

def show_home():
    st.markdown('<div class="section-header">Welcome to Your Analytics Platform!</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This Platform Does:
        
        **1. 📤 Data Upload**
        - Upload CSV, Excel, or JSON files
        - Automatic data type detection
        - Preview your dataset instantly
        
        **2. 🔍 Exploratory Data Analysis**
        - Automated statistical summaries
        - Distribution analysis
        - Correlation matrices
        - Missing value analysis
        
        **3. 🧹 Data Cleaning**
        - Handle missing values intelligently
        - Remove duplicates
        - Detect and handle outliers
        - Data type conversions
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Advanced Features:
        
        **4. 💬 Natural Language SQL**
        - Ask questions in plain English
        - Get SQL queries automatically
        - Execute queries on your data
        
        **5. 📊 Interactive Visualizations**
        - Multiple chart types
        - Interactive plotly charts
        - Customizable dashboards
        
        **6. 🤖 Machine Learning**
        - Automated model selection
        - Classification & Regression
        - Model performance metrics
        - Prediction capabilities
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate through different sections. Start by uploading your dataset!")

def data_upload_section():
    st.markdown('<div class="section-header">📤 Data Upload Section</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your dataset file",
        type=['csv', 'xlsx', 'json'],
        help="Upload CSV, Excel, or JSON files"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            # Store in session state
            st.session_state.data = df
            
            st.success(f"✅ Successfully loaded {uploaded_file.name}!")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", df.shape[0])
            with col2:
                st.metric("📋 Columns", df.shape[1])
            with col3:
                st.metric("💾 Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show column information
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to get started!")

def eda_section():
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.data
    
    # EDA Options
    eda_option = st.selectbox(
        "Select EDA Analysis:",
        ["Statistical Summary", "Distribution Analysis", "Correlation Analysis", "Missing Value Analysis"]
    )
    
    if eda_option == "Statistical Summary":
        st.subheader("📈 Statistical Summary")
        
        # Numerical columns summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.write("**Numerical Columns:**")
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.write("**Categorical Columns:**")
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                st.write(f"**{col}:**")
                value_counts = df[col].value_counts().head(10)
                st.bar_chart(value_counts)
    
    elif eda_option == "Distribution Analysis":
        st.subheader("📊 Distribution Analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            selected_col = st.selectbox("Select column for distribution:", numerical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
    
    elif eda_option == "Correlation Analysis":
        st.subheader("🔗 Correlation Analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            
            # Plotly heatmap
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numerical columns for correlation analysis.")
    
    elif eda_option == "Missing Value Analysis":
        st.subheader("❓ Missing Value Analysis")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualization
            fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                        title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")

def data_cleaning_section():
    st.markdown('<div class="section-header">🧹 Data Cleaning Section</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.data.copy()
    
    st.subheader("🔧 Cleaning Options")
    
    # Missing value handling
    st.write("**1. Handle Missing Values:**")
    missing_strategy = st.selectbox(
        "Select strategy for missing values:",
        ["Keep as is", "Drop rows with missing values", "Fill with mean/mode", "Fill with median"]
    )
    
    # Duplicate handling
    st.write("**2. Handle Duplicates:**")
    handle_duplicates = st.checkbox("Remove duplicate rows")
    
    # Outlier handling
    st.write("**3. Handle Outliers:**")
    handle_outliers = st.checkbox("Remove outliers (using IQR method)")
    
    if st.button("🚀 Apply Cleaning"):
        cleaned_df = df.copy()
        
        # Handle missing values
        if missing_strategy == "Drop rows with missing values":
            cleaned_df = cleaned_df.dropna()
            st.info(f"Dropped {len(df) - len(cleaned_df)} rows with missing values")
        elif missing_strategy == "Fill with mean/mode":
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                else:
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown', inplace=True)
            st.info("Filled missing values with mean/mode")
        elif missing_strategy == "Fill with median":
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                else:
                    cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown', inplace=True)
            st.info("Filled missing values with median/mode")
        
        # Handle duplicates
        if handle_duplicates:
            before_dup = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            st.info(f"Removed {before_dup - len(cleaned_df)} duplicate rows")
        
        # Handle outliers
        if handle_outliers:
            numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                before_outlier = len(cleaned_df)
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
                st.info(f"Removed {before_outlier - len(cleaned_df)} outliers from {col}")
        
        # Store cleaned data
        st.session_state.cleaned_data = cleaned_df
        
        # Show comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", len(df))
        with col2:
            st.metric("Cleaned Rows", len(cleaned_df))
        
        st.success("✅ Data cleaning completed!")
        st.dataframe(cleaned_df.head(), use_container_width=True)

def natural_language_sql():
    st.markdown('<div class="section-header">💬 Natural Language to SQL</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    
    st.write("Ask questions about your data in plain English!")
    
    # Sample questions
    st.subheader("💡 Example Questions:")
    examples = [
        "Show me the top 10 records",
        "What is the average of [column_name]?",
        "Count the number of unique values in [column_name]",
        "Show records where [column_name] is greater than 100",
        "Group by [column_name] and show counts"
    ]
    
    for example in examples:
        st.code(example)
    
    user_question = st.text_input("🤔 Your Question:", placeholder="e.g., Show me records where age is greater than 25")
    
    if st.button("🔍 Generate SQL & Execute"):
        if user_question:
            # Simple rule-based SQL generation (you can enhance this with OpenAI API)
            sql_query = generate_sql_from_question(user_question, df.columns.tolist())
            
            if sql_query:
                st.subheader("🔧 Generated SQL:")
                st.code(sql_query, language='sql')
                
                try:
                    # Execute the query using pandas (simulating SQL)
                    result = execute_pandas_query(df, user_question)
                    
                    st.subheader("📊 Results:")
                    if result is not None:
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.error("Could not execute the query")
                        
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
            else:
                st.error("Could not generate SQL from your question. Try rephrasing!")

def generate_sql_from_question(question, columns):
    """Simple rule-based SQL generation - you can replace this with OpenAI API"""
    question_lower = question.lower()
    
    if "top" in question_lower and any(char.isdigit() for char in question):
        number = ''.join(filter(str.isdigit, question))
        return f"SELECT * FROM dataset LIMIT {number};"
    
    elif "average" in question_lower or "avg" in question_lower:
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT AVG({col}) FROM dataset;"
    
    elif "count" in question_lower:
        for col in columns:
            if col.lower() in question_lower:
                return f"SELECT COUNT(DISTINCT {col}) FROM dataset;"
    
    elif "greater than" in question_lower:
        # Extract column and value
        parts = question_lower.split("greater than")
        if len(parts) == 2:
            value = ''.join(filter(str.isdigit, parts[1]))
            for col in columns:
                if col.lower() in parts[0]:
                    return f"SELECT * FROM dataset WHERE {col} > {value};"
    
    return None

def execute_pandas_query(df, question):
    """Execute pandas equivalent of SQL queries"""
    question_lower = question.lower()
    
    if "top" in question_lower and any(char.isdigit() for char in question):
        number = int(''.join(filter(str.isdigit, question)))
        return df.head(number)
    
    elif "average" in question_lower or "avg" in question_lower:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col.lower() in question_lower:
                return pd.DataFrame({f'Average of {col}': [df[col].mean()]})
    
    elif "count" in question_lower:
        for col in df.columns:
            if col.lower() in question_lower:
                return pd.DataFrame({f'Unique count of {col}': [df[col].nunique()]})
    
    elif "greater than" in question_lower:
        parts = question_lower.split("greater than")
        if len(parts) == 2:
            try:
                value = float(''.join(filter(lambda x: x.isdigit() or x == '.', parts[1])))
                for col in df.columns:
                    if col.lower() in parts[0] and df[col].dtype in ['int64', 'float64']:
                        return df[df[col] > value]
            except:
                pass
    
    return df.head()  # Default return

def visualization_section():
    st.markdown('<div class="section-header">📊 Advanced Visualization Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    
    # Import the advanced visualization module (you'll need to have advanced_viz.py in the same directory)
    try:
        from advanced_viz import PowerBIStyleDashboard, ReportGenerator
        
        dashboard = PowerBIStyleDashboard(df)
        
        # Dashboard type selection
        dashboard_type = st.selectbox(
            "Select Dashboard Type:",
            ["Executive Summary", "Detailed Analysis", "Interactive Filters", "Dynamic Charts", "Multi-Chart Dashboard"]
        )
        
        if dashboard_type == "Executive Summary":
            dashboard.create_executive_summary()
        
        elif dashboard_type == "Detailed Analysis":
            dashboard.create_detailed_analysis()
        
        elif dashboard_type == "Interactive Filters":
            filtered_df = dashboard.create_interactive_filters()
            st.session_state.filtered_data = filtered_df
        
        elif dashboard_type == "Dynamic Charts":
            dashboard.create_dynamic_chart()
        
        elif dashboard_type == "Multi-Chart Dashboard":
            dashboard.create_multi_chart_dashboard()
        
        # Report Generation Section
        st.markdown("---")
        st.subheader("📋 Generate Comprehensive Report")
        
        if st.button("🚀 Generate Analysis Report"):
            report_gen = ReportGenerator(df)
            report_content = report_gen.generate_summary_report()
            
            st.markdown("### 📊 Generated Report:")
            st.markdown(report_content)
            
            # Export options
            report_gen.export_report_options(report_content)
    
    except ImportError:
        st.error("Advanced visualization module not found. Using basic visualization.")
        # Fall back to basic visualization
        basic_visualization_fallback(df)

def machine_learning_section():
    st.markdown('<div class="section-header">🤖 Machine Learning Section</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
    
    # ML Task Selection
    ml_task = st.selectbox(
        "Select ML Task:",
        ["Classification", "Regression"]
    )
    
    # Target variable selection
    target_column = st.selectbox("Select Target Column:", df.columns)
    
    # Feature selection
    available_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect("Select Feature Columns:", available_features, default=available_features[:5])
    
    if len(selected_features) > 0 and target_column:
        if st.button("🚀 Train Model"):
            try:
                # Prepare data
                X = df[selected_features]
                y = df[target_column]
                
                # Handle categorical variables in features
                X_processed = X.copy()
                for col in X_processed.columns:
                    if X_processed[col].dtype == 'object':
                        X_processed[col] = pd.factorize(X_processed[col])[0]
                
                # Handle categorical target for classification
                if ml_task == "Classification" and y.dtype == 'object':
                    y = pd.factorize(y)[0]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
                
                # Train model
                if ml_task == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    
                    st.success(f"✅ Classification Model Trained!")
                    st.metric("Accuracy", f"{accuracy:.4f}")
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                
                elif ml_task == "Regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    
                    st.success(f"✅ Regression Model Trained!")
                    st.metric("RMSE", f"{rmse:.4f}")
                    
                    # Actual vs Predicted plot
                    fig = px.scatter(x=y_test, y=predictions, 
                                   labels={'x': 'Actual', 'y': 'Predicted'},
                                   title='Actual vs Predicted Values')
                    fig.add_shape(type='line', x0=min(y_test), y0=min(y_test), 
                                x1=max(y_test), y1=max(y_test), line=dict(color='red'))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                               title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
    
    else:
        st.info("👆 Please select target column and at least one feature column.")

if __name__ == "__main__":
    main()