# nlp_to_sql.py - Advanced Natural Language to SQL Converter
import re
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Tuple

class NLPToSQL:
    """
    Advanced Natural Language to SQL converter
    This class can be enhanced with OpenAI API for better accuracy
    """
    
    def __init__(self, dataframe: pd.DataFrame, table_name: str = "dataset"):
        self.df = dataframe
        self.table_name = table_name
        self.columns = dataframe.columns.tolist()
        self.numeric_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
        
    def preprocess_question(self, question: str) -> str:
        """Clean and normalize the input question"""
        question = question.lower().strip()
        # Remove common question words that don't affect SQL
        question = re.sub(r'\b(what|how|can you|please|show me|tell me|give me)\b', '', question)
        return question.strip()
    
    def identify_columns(self, question: str) -> List[str]:
        """Identify which columns are mentioned in the question"""
        identified_cols = []
        question_lower = question.lower()
        
        for col in self.columns:
            col_variations = [
                col.lower(),
                col.lower().replace('_', ' '),
                col.lower().replace(' ', '_'),
                col.lower().replace('-', ' ')
            ]
            
            for variation in col_variations:
                if variation in question_lower:
                    identified_cols.append(col)
                    break
        
        return list(set(identified_cols))
    
    def extract_numbers(self, question: str) -> List[float]:
        """Extract numeric values from the question"""
        numbers = re.findall(r'-?\d+\.?\d*', question)
        return [float(num) for num in numbers]
    
    def identify_operation(self, question: str) -> str:
        """Identify the main operation requested"""
        question_lower = question.lower()
        
        # Aggregation operations
        if any(word in question_lower for word in ['average', 'avg', 'mean']):
            return 'average'
        elif any(word in question_lower for word in ['sum', 'total', 'add up']):
            return 'sum'
        elif any(word in question_lower for word in ['count', 'number of', 'how many']):
            return 'count'
        elif any(word in question_lower for word in ['maximum', 'max', 'highest', 'largest']):
            return 'max'
        elif any(word in question_lower for word in ['minimum', 'min', 'lowest', 'smallest']):
            return 'min'
        
        # Filter operations
        elif any(word in question_lower for word in ['greater than', 'more than', '>', 'above']):
            return 'filter_greater'
        elif any(word in question_lower for word in ['less than', 'fewer than', '<', 'below']):
            return 'filter_less'
        elif any(word in question_lower for word in ['equal to', 'equals', '=']):
            return 'filter_equal'
        
        # Sorting operations
        elif any(word in question_lower for word in ['top', 'highest', 'largest', 'biggest']):
            return 'top'
        elif any(word in question_lower for word in ['bottom', 'lowest', 'smallest']):
            return 'bottom'
        
        # Group by operations
        elif any(word in question_lower for word in ['group by', 'grouped by', 'by each', 'for each']):
            return 'group_by'
        
        # Default select
        return 'select'
    
    def generate_sql(self, question: str) -> Optional[str]:
        """Generate SQL query from natural language question"""
        try:
            processed_question = self.preprocess_question(question)
            columns = self.identify_columns(processed_question)
            numbers = self.extract_numbers(processed_question)
            operation = self.identify_operation(processed_question)
            
            # Base SQL components
            select_clause = "*"
            from_clause = f"FROM {self.table_name}"
            where_clause = ""
            group_by_clause = ""
            order_by_clause = ""
            limit_clause = ""
            
            if operation == 'average' and columns:
                select_clause = f"AVG({columns[0]}) as average_{columns[0]}"
            
            elif operation == 'sum' and columns:
                select_clause = f"SUM({columns[0]}) as sum_{columns[0]}"
            
            elif operation == 'count':
                if columns:
                    select_clause = f"COUNT(DISTINCT {columns[0]}) as count_{columns[0]}"
                else:
                    select_clause = "COUNT(*) as total_count"
            
            elif operation == 'max' and columns:
                select_clause = f"MAX({columns[0]}) as max_{columns[0]}"
            
            elif operation == 'min' and columns:
                select_clause = f"MIN({columns[0]}) as min_{columns[0]}"
            
            elif operation == 'filter_greater' and columns and numbers:
                where_clause = f"WHERE {columns[0]} > {numbers[0]}"
            
            elif operation == 'filter_less' and columns and numbers:
                where_clause = f"WHERE {columns[0]} < {numbers[0]}"
            
            elif operation == 'filter_equal' and columns and numbers:
                where_clause = f"WHERE {columns[0]} = {numbers[0]}"
            
            elif operation == 'top' and numbers:
                limit_num = int(numbers[0])
                if columns and columns[0] in self.numeric_columns:
                    order_by_clause = f"ORDER BY {columns[0]} DESC"
                limit_clause = f"LIMIT {limit_num}"
            
            elif operation == 'bottom' and numbers:
                limit_num = int(numbers[0])
                if columns and columns[0] in self.numeric_columns:
                    order_by_clause = f"ORDER BY {columns[0]} ASC"
                limit_clause = f"LIMIT {limit_num}"
            
            elif operation == 'group_by' and columns:
                if len(columns) >= 2:
                    select_clause = f"{columns[0]}, COUNT(*) as count"
                    group_by_clause = f"GROUP BY {columns[0]}"
                elif len(columns) == 1:
                    select_clause = f"{columns[0]}, COUNT(*) as count"
                    group_by_clause = f"GROUP BY {columns[0]}"
            
            # Construct final SQL
            sql_parts = [f"SELECT {select_clause}", from_clause]
            
            if where_clause:
                sql_parts.append(where_clause)
            if group_by_clause:
                sql_parts.append(group_by_clause)
            if order_by_clause:
                sql_parts.append(order_by_clause)
            if limit_clause:
                sql_parts.append(limit_clause)
            
            return " ".join(sql_parts) + ";"
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None
    
    def execute_pandas_equivalent(self, question: str) -> Optional[pd.DataFrame]:
        """Execute pandas equivalent of the SQL query"""
        try:
            processed_question = self.preprocess_question(question)
            columns = self.identify_columns(processed_question)
            numbers = self.extract_numbers(processed_question)
            operation = self.identify_operation(processed_question)
            
            df_result = self.df.copy()
            
            if operation == 'average' and columns and columns[0] in self.numeric_columns:
                result = df_result[columns[0]].mean()
                return pd.DataFrame({f'average_{columns[0]}': [result]})
            
            elif operation == 'sum' and columns and columns[0] in self.numeric_columns:
                result = df_result[columns[0]].sum()
                return pd.DataFrame({f'sum_{columns[0]}': [result]})
            
            elif operation == 'count':
                if columns and columns[0] in df_result.columns:
                    result = df_result[columns[0]].nunique()
                    return pd.DataFrame({f'count_unique_{columns[0]}': [result]})
                else:
                    result = len(df_result)
                    return pd.DataFrame({'total_count': [result]})
            
            elif operation == 'max' and columns and columns[0] in self.numeric_columns:
                result = df_result[columns[0]].max()
                return pd.DataFrame({f'max_{columns[0]}': [result]})
            
            elif operation == 'min' and columns and columns[0] in self.numeric_columns:
                result = df_result[columns[0]].min()
                return pd.DataFrame({f'min_{columns[0]}': [result]})
            
            elif operation == 'filter_greater' and columns and numbers:
                if columns[0] in self.numeric_columns:
                    return df_result[df_result[columns[0]] > numbers[0]]
            
            elif operation == 'filter_less' and columns and numbers:
                if columns[0] in self.numeric_columns:
                    return df_result[df_result[columns[0]] < numbers[0]]
            
            elif operation == 'filter_equal' and columns and numbers:
                if columns[0] in self.numeric_columns:
                    return df_result[df_result[columns[0]] == numbers[0]]
                else:
                    return df_result[df_result[columns[0]] == str(numbers[0])]
            
            elif operation == 'top' and numbers:
                limit_num = int(numbers[0])
                if columns and columns[0] in self.numeric_columns:
                    return df_result.nlargest(limit_num, columns[0])
                else:
                    return df_result.head(limit_num)
            
            elif operation == 'bottom' and numbers:
                limit_num = int(numbers[0])
                if columns and columns[0] in self.numeric_columns:
                    return df_result.nsmallest(limit_num, columns[0])
                else:
                    return df_result.tail(limit_num)
            
            elif operation == 'group_by' and columns:
                if columns[0] in df_result.columns:
                    return df_result.groupby(columns[0]).size().reset_index(name='count')
            
            # Default: return first few rows
            return df_result.head(10)
            
        except Exception as e:
            print(f"Error executing pandas equivalent: {e}")
            return None

# Advanced SQL Query Templates
class SQLTemplateGenerator:
    """Generate more complex SQL queries"""
    
    @staticmethod
    def generate_analytics_queries(table_name: str, columns: List[str]) -> Dict[str, str]:
        """Generate common analytics SQL queries"""
        numeric_cols = [col for col in columns if col.lower() in ['age', 'price', 'salary', 'revenue', 'amount', 'value', 'score']]
        categorical_cols = [col for col in columns if col.lower() in ['category', 'type', 'status', 'region', 'department']]
        
        queries = {
            "basic_stats": f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT {columns[0]}) as unique_{columns[0]}
                FROM {table_name};
            """,
            
            "missing_data_analysis": f"""
                SELECT 
                    {', '.join([f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as missing_{col}" for col in columns[:5]])}
                FROM {table_name};
            """
        }
        
        if numeric_cols:
            col = numeric_cols[0]
            queries["descriptive_stats"] = f"""
                SELECT 
                    MIN({col}) as min_{col},
                    MAX({col}) as max_{col},
                    AVG({col}) as avg_{col},
                    COUNT({col}) as count_{col}
                FROM {table_name};
            """
        
        if categorical_cols:
            col = categorical_cols[0]
            queries["category_distribution"] = f"""
                SELECT 
                    {col},
                    COUNT(*) as frequency,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}), 2) as percentage
                FROM {table_name}
                GROUP BY {col}
                ORDER BY frequency DESC;
            """
        
        return queries