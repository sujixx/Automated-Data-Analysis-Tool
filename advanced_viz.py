# advanced_viz.py - Power BI-like Visualization Module
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class PowerBIStyleDashboard:
    """
    Create Power BI-style interactive dashboards
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
        self.date_cols = dataframe.select_dtypes(include=['datetime64']).columns.tolist()
    
    def create_kpi_cards(self) -> dict:
        """Create KPI cards similar to Power BI"""
        kpis = {}
        
        # Basic KPIs
        kpis['Total Records'] = len(self.df)
        kpis['Total Columns'] = len(self.df.columns)
        kpis['Missing Values'] = self.df.isnull().sum().sum()
        kpis['Complete Records'] = len(self.df.dropna())
        
        # Numeric KPIs
        if self.numeric_cols:
            for col in self.numeric_cols[:3]:  # Top 3 numeric columns
                kpis[f'Avg {col}'] = round(self.df[col].mean(), 2)
                kpis[f'Max {col}'] = round(self.df[col].max(), 2)
                kpis[f'Min {col}'] = round(self.df[col].min(), 2)
        
        return kpis
    
    def create_executive_summary(self):
        """Create an executive summary dashboard"""
        st.subheader("📊 Executive Dashboard")
        
        # KPI Cards
        kpis = self.create_kpi_cards()
        
        # Display KPIs in columns
        num_cols = 4
        cols = st.columns(num_cols)
        
        for i, (kpi_name, kpi_value) in enumerate(kpis.items()):
            with cols[i % num_cols]:
                st.metric(label=kpi_name, value=kpi_value)
        
        # Main charts
        col1, col2 = st.columns(2)
        
        with col1:
            if self.numeric_cols:
                # Distribution chart
                selected_numeric = st.selectbox("Select Numeric Column:", self.numeric_cols, key="exec_numeric")
                fig = px.histogram(self.df, x=selected_numeric, 
                                 title=f"Distribution of {selected_numeric}",
                                 color_discrete_sequence=['#1f77b4'])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if self.categorical_cols:
                # Categorical distribution
                selected_cat = st.selectbox("Select Categorical Column:", self.categorical_cols, key="exec_cat")
                value_counts = self.df[selected_cat].value_counts().head(10)
                
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {selected_cat}")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    def create_detailed_analysis(self):
        """Create detailed analysis dashboard"""
        st.subheader("🔍 Detailed Analysis Dashboard")
        
        # Multi-select for analysis type
        analysis_types = st.multiselect(
            "Select Analysis Types:",
            ["Correlation Analysis", "Trend Analysis", "Comparative Analysis", "Statistical Overview"],
            default=["Correlation Analysis"]
        )
        
        if "Correlation Analysis" in analysis_types and len(self.numeric_cols) > 1:
            st.write("**📈 Correlation Analysis**")
            
            # Correlation heatmap
            corr_matrix = self.df[self.numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                          text_auto=True,
                          aspect="auto",
                          color_continuous_scale="RdBu",
                          title="Correlation Matrix")
            
            fig.update_layout(
                title_x=0.5,
                width=800,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if "Trend Analysis" in analysis_types and self.numeric_cols:
            st.write("**📊 Trend Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                y_column = st.selectbox("Select Y-axis:", self.numeric_cols, key="trend_y")
            with col2:
                if len(self.numeric_cols) > 1:
                    x_column = st.selectbox("Select X-axis:", self.numeric_cols, key="trend_x")
                else:
                    x_column = None
            
            if x_column and x_column != y_column:
                fig = px.scatter(self.df, x=x_column, y=y_column,
                               title=f"{y_column} vs {x_column}",
                               trendline="ols")
                fig.update_layout(title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
        
        if "Comparative Analysis" in analysis_types and self.categorical_cols and self.numeric_cols:
            st.write("**⚖️ Comparative Analysis**")
            
            col1, col2 = st.columns(2)
            with col1:
                category_col = st.selectbox("Select Category:", self.categorical_cols, key="comp_cat")
            with col2:
                numeric_col = st.selectbox("Select Numeric Value:", self.numeric_cols, key="comp_num")
            
            # Box plot for comparison
            fig = px.box(self.df, x=category_col, y=numeric_col,
                        title=f"{numeric_col} by {category_col}")
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        if "Statistical Overview" in analysis_types:
            st.write("**📋 Statistical Overview**")
            
            # Create a comprehensive statistical table
            stats_data = []
            
            for col in self.numeric_cols:
                stats_data.append({
                    'Column': col,
                    'Mean': round(self.df[col].mean(), 2),
                    'Median': round(self.df[col].median(), 2),
                    'Std Dev': round(self.df[col].std(), 2),
                    'Min': round(self.df[col].min(), 2),
                    'Max': round(self.df[col].max(), 2),
                    'Missing %': round((self.df[col].isnull().sum() / len(self.df)) * 100, 2)
                })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
    
    def create_interactive_filters(self):
        """Create interactive filters similar to Power BI slicers"""
        st.subheader("🎛️ Interactive Filters")
        
        filtered_df = self.df.copy()
        
        # Create filters for categorical columns
        filters = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if self.categorical_cols:
                st.write("**Categorical Filters:**")
                for cat_col in self.categorical_cols[:3]:  # Limit to first 3
                    unique_values = self.df[cat_col].unique()
                    selected_values = st.multiselect(
                        f"Filter by {cat_col}:",
                        options=unique_values,
                        default=unique_values,
                        key=f"filter_{cat_col}"
                    )
                    if selected_values:
                        filters[cat_col] = selected_values
        
        with col2:
            if self.numeric_cols:
                st.write("**Numeric Range Filters:**")
                for num_col in self.numeric_cols[:2]:  # Limit to first 2
                    min_val = float(self.df[num_col].min())
                    max_val = float(self.df[num_col].max())
                    
                    range_values = st.slider(
                        f"Range for {num_col}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"range_{num_col}"
                    )
                    filters[f"{num_col}_range"] = range_values
        
        # Apply filters
        for filter_col, filter_values in filters.items():
            if filter_col.endswith('_range'):
                col_name = filter_col.replace('_range', '')
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= filter_values[0]) & 
                    (filtered_df[col_name] <= filter_values[1])
                ]
            else:
                filtered_df = filtered_df[filtered_df[filter_col].isin(filter_values)]
        
        # Show filtered results
        st.write(f"**Filtered Dataset: {len(filtered_df)} records (from {len(self.df)} total)**")
        
        if len(filtered_df) > 0:
            # Quick stats on filtered data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Filtered Records", len(filtered_df))
            with col2:
                if self.numeric_cols:
                    avg_val = filtered_df[self.numeric_cols[0]].mean()
                    st.metric(f"Avg {self.numeric_cols[0]}", f"{avg_val:.2f}")
            with col3:
                missing_pct = (filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100
                st.metric("Missing Data %", f"{missing_pct:.1f}%")
            
            # Show preview of filtered data
            st.dataframe(filtered_df.head(10), use_container_width=True)
            
            # Create visualization with filtered data
            if st.button("📊 Create Chart with Filtered Data"):
                self.create_dynamic_chart(filtered_df)
        
        return filtered_df
    
    def create_dynamic_chart(self, df=None):
        """Create dynamic charts based on user selection"""
        if df is None:
            df = self.df
        
        st.subheader("📈 Dynamic Chart Builder")
        
        # Chart type selection
        chart_types = [
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
            "Box Plot", "Violin Plot", "Heatmap", "3D Scatter"
        ]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox("Chart Type:", chart_types)
        
        with col2:
            if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart", "3D Scatter"]:
                x_axis = st.selectbox("X-axis:", df.columns, key="dynamic_x")
        
        with col3:
            if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Violin Plot", "3D Scatter"]:
                y_axis = st.selectbox("Y-axis:", df.columns, key="dynamic_y")
        
        # Additional options
        color_by = None
        size_by = None
        
        if chart_type in ["Scatter Plot", "3D Scatter"]:
            col1, col2 = st.columns(2)
            with col1:
                if self.categorical_cols:
                    color_by = st.selectbox("Color by:", ["None"] + self.categorical_cols, key="color_by")
                    color_by = None if color_by == "None" else color_by
            with col2:
                if self.numeric_cols:
                    size_by = st.selectbox("Size by:", ["None"] + self.numeric_cols, key="size_by")
                    size_by = None if size_by == "None" else size_by
        
        # Generate chart
        try:
            if chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, size=size_by,
                               title=f"{y_axis} vs {x_axis}")
            
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            
            elif chart_type == "Bar Chart":
                if df[x_axis].dtype == 'object':
                    # Aggregate for categorical x-axis
                    agg_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis, title=f"Average {y_axis} by {x_axis}")
                else:
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            
            elif chart_type == "Histogram":
                selected_col = st.selectbox("Select Column:", df.columns, key="hist_col")
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            
            elif chart_type == "Box Plot":
                fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            
            elif chart_type == "Violin Plot":
                fig = px.violin(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            
            elif chart_type == "Heatmap":
                numeric_data = df.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                   title="Correlation Heatmap")
                else:
                    st.error("Need at least 2 numeric columns for heatmap")
                    return
            
            elif chart_type == "3D Scatter":
                if len(self.numeric_cols) >= 3:
                    z_axis = st.selectbox("Z-axis:", self.numeric_cols, key="z_axis")
                    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=color_by,
                                       title=f"3D Plot: {x_axis} vs {y_axis} vs {z_axis}")
                else:
                    st.error("Need at least 3 numeric columns for 3D scatter plot")
                    return
            
            # Enhance chart appearance
            fig.update_layout(
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.write("**📁 Export Options:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Chart as HTML"):
                    html_str = fig.to_html()
                    st.download_button(
                        label="Download HTML",
                        data=html_str,
                        file_name="chart.html",
                        mime="text/html"
                    )
            
            with col2:
                if st.button("📈 Download Chart as JSON"):
                    json_str = fig.to_json()
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="chart.json",
                        mime="application/json"
                    )
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    
    def create_multi_chart_dashboard(self):
        """Create a dashboard with multiple charts"""
        st.subheader("📊 Multi-Chart Dashboard")
        
        # Layout selection
        layout = st.radio("Select Layout:", ["2x1 (Side by Side)", "1x2 (Stacked)", "2x2 (Grid)"])
        
        if layout == "2x1 (Side by Side)":
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Chart 1:**")
                if self.numeric_cols:
                    fig1 = px.histogram(self.df, x=self.numeric_cols[0], 
                                       title=f"Distribution of {self.numeric_cols[0]}")
                    fig1.update_layout(height=400)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.write("**Chart 2:**")
                if self.categorical_cols:
                    value_counts = self.df[self.categorical_cols[0]].value_counts().head(8)
                    fig2 = px.pie(values=value_counts.values, names=value_counts.index,
                                 title=f"Distribution of {self.categorical_cols[0]}")
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
        
        elif layout == "1x2 (Stacked)":
            st.write("**Chart 1:**")
            if len(self.numeric_cols) >= 2:
                fig1 = px.scatter(self.df, x=self.numeric_cols[0], y=self.numeric_cols[1],
                                 title=f"{self.numeric_cols[1]} vs {self.numeric_cols[0]}")
                st.plotly_chart(fig1, use_container_width=True)
            
            st.write("**Chart 2:**")
            if len(self.numeric_cols) > 1:
                corr_matrix = self.df[self.numeric_cols].corr()
                fig2 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Correlation Matrix")
                st.plotly_chart(fig2, use_container_width=True)
        
        elif layout == "2x2 (Grid)":
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Left:**")
                if self.numeric_cols:
                    fig1 = px.histogram(self.df, x=self.numeric_cols[0],
                                       title=f"Distribution of {self.numeric_cols[0]}")
                    fig1.update_layout(height=300)
                    st.plotly_chart(fig1, use_container_width=True)
                
                st.write("**Bottom Left:**")
                if len(self.numeric_cols) >= 2:
                    fig3 = px.box(self.df, y=self.numeric_cols[1],
                                 title=f"Box Plot of {self.numeric_cols[1]}")
                    fig3.update_layout(height=300)
                    st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                st.write("**Top Right:**")
                if self.categorical_cols:
                    value_counts = self.df[self.categorical_cols[0]].value_counts().head(6)
                    fig2 = px.bar(x=value_counts.index, y=value_counts.values,
                                 title=f"Count of {self.categorical_cols[0]}")
                    fig2.update_layout(height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.write("**Bottom Right:**")
                if len(self.numeric_cols) >= 2:
                    fig4 = px.scatter(self.df, x=self.numeric_cols[0], y=self.numeric_cols[1],
                                     title="Scatter Plot")
                    fig4.update_layout(height=300)
                    st.plotly_chart(fig4, use_container_width=True)

class ReportGenerator:
    """Generate comprehensive reports similar to Power BI reports"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        
        report_sections = []
        
        # Executive Summary
        report_sections.append("# 📊 Data Analysis Report")
        report_sections.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append("\n## 📈 Executive Summary")
        report_sections.append(f"- **Total Records:** {len(self.df):,}")
        report_sections.append(f"- **Total Columns:** {len(self.df.columns)}")
        report_sections.append(f"- **Numeric Columns:** {len(self.numeric_cols)}")
        report_sections.append(f"- **Categorical Columns:** {len(self.categorical_cols)}")
        report_sections.append(f"- **Missing Values:** {self.df.isnull().sum().sum():,}")
        report_sections.append(f"- **Data Completeness:** {((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100):.1f}%")
        
        # Data Quality Assessment
        report_sections.append("\n## 🔍 Data Quality Assessment")
        
        # Missing values by column
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            report_sections.append("### Missing Values by Column:")
            for col, missing_count in missing_data.head(10).items():
                missing_pct = (missing_count / len(self.df)) * 100
                report_sections.append(f"- **{col}:** {missing_count:,} ({missing_pct:.1f}%)")
        else:
            report_sections.append("✅ **No missing values found!**")
        
        # Duplicate rows
        duplicate_count = self.df.duplicated().sum()
        report_sections.append(f"\n### Duplicate Rows: {duplicate_count:,}")
        
        # Statistical Summary for Numeric Columns
        if self.numeric_cols:
            report_sections.append("\n## 📊 Statistical Summary")
            
            for col in self.numeric_cols[:5]:  # Top 5 numeric columns
                report_sections.append(f"\n### {col}")
                report_sections.append(f"- **Mean:** {self.df[col].mean():.2f}")
                report_sections.append(f"- **Median:** {self.df[col].median():.2f}")
                report_sections.append(f"- **Standard Deviation:** {self.df[col].std():.2f}")
                report_sections.append(f"- **Min:** {self.df[col].min():.2f}")
                report_sections.append(f"- **Max:** {self.df[col].max():.2f}")
        
        # Categorical Summary
        if self.categorical_cols:
            report_sections.append("\n## 📝 Categorical Analysis")
            
            for col in self.categorical_cols[:5]:  # Top 5 categorical columns
                unique_count = self.df[col].nunique()
                most_common = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "N/A"
                
                report_sections.append(f"\n### {col}")
                report_sections.append(f"- **Unique Values:** {unique_count}")
                report_sections.append(f"- **Most Common:** {most_common}")
                
                # Top categories
                top_categories = self.df[col].value_counts().head(5)
                report_sections.append("- **Top Categories:**")
                for category, count in top_categories.items():
                    pct = (count / len(self.df)) * 100
                    report_sections.append(f"  - {category}: {count:,} ({pct:.1f}%)")
        
        # Recommendations
        report_sections.append("\n## 💡 Recommendations")
        
        recommendations = []
        
        # Data quality recommendations
        if self.df.isnull().sum().sum() > 0:
            recommendations.append("🧹 **Data Cleaning:** Consider handling missing values through imputation or removal")
        
        if duplicate_count > 0:
            recommendations.append("🔄 **Duplicate Handling:** Remove or investigate duplicate records")
        
        # Analysis recommendations
        if len(self.numeric_cols) > 1:
            recommendations.append("📈 **Correlation Analysis:** Explore relationships between numeric variables")
        
        if self.categorical_cols and self.numeric_cols:
            recommendations.append("📊 **Segmentation Analysis:** Analyze numeric variables by categorical segments")
        
        if len(recommendations) == 0:
            recommendations.append("✅ **Data Quality:** Your dataset appears to be in good condition!")
        
        report_sections.extend(recommendations)
        
        return "\n".join(report_sections)
    
    def export_report_options(self, report_content: str):
        """Provide export options for the report"""
        st.write("**📁 Export Report:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download as Text",
                data=report_content,
                file_name="data_analysis_report.txt",
                mime="text/plain"
            )
        
        with col2:
            # Convert to HTML for better formatting
            html_content = report_content.replace('\n', '<br>')
            html_content = f"<html><body><pre>{html_content}</pre></body></html>"
            
            st.download_button(
                label="🌐 Download as HTML",
                data=html_content,
                file_name="data_analysis_report.html",
                mime="text/html"
            )
        
        with col3:
            # Create a simple CSV summary
            summary_data = {
                'Metric': ['Total Records', 'Total Columns', 'Missing Values', 'Duplicate Records'],
                'Value': [len(self.df), len(self.df.columns), self.df.isnull().sum().sum(), self.df.duplicated().sum()]
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.download_button(
                label="📊 Download Summary CSV",
                data=summary_df.to_csv(index=False),
                file_name="data_summary.csv",
                mime="text/csv"
            )