"""
Data Agent - Data processing, analysis, and visualization
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import io
import base64
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from sources.agents.agent import Agent
from sources.logger import Logger


class DataAgent(Agent):
    """
    Specialized agent for data manipulation and analysis
    """
    
    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)
        self.logger = Logger("data_agent.log")
        self.dataframes = {}  # Store loaded dataframes
        
    def generate_response(self, query: str) -> Tuple[str, str]:
        """
        Main entry point for data-related requests
        """
        self.logger.log(f"Data query: {query}")
        
        # Determine data operation type
        operation = self._identify_operation(query)
        
        if operation == "load":
            response = self._handle_load_data(query)
        elif operation == "analyze":
            response = self._handle_analyze_data(query)
        elif operation == "visualize":
            response = self._handle_visualize_data(query)
        elif operation == "transform":
            response = self._handle_transform_data(query)
        elif operation == "export":
            response = self._handle_export_data(query)
        else:
            response = self._handle_general_data_query(query)
            
        return response, "data"
        
    def _identify_operation(self, query: str) -> str:
        """Identify the type of data operation requested"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["load", "read", "import", "open"]):
            return "load"
        elif any(word in query_lower for word in ["analyze", "statistics", "summary", "describe"]):
            return "analyze"
        elif any(word in query_lower for word in ["plot", "chart", "graph", "visualize", "show"]):
            return "visualize"
        elif any(word in query_lower for word in ["transform", "clean", "filter", "merge", "group"]):
            return "transform"
        elif any(word in query_lower for word in ["export", "save", "write"]):
            return "export"
        else:
            return "general"
            
    def load_data(self, file_path: str, file_type: str = None) -> pd.DataFrame:
        """Load data from file"""
        try:
            if file_type == "csv" or file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_type == "excel" or file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_type == "json" or file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Store dataframe with file name as key
            df_name = file_path.split('/')[-1].split('.')[0]
            self.dataframes[df_name] = df
            
            self.logger.log(f"Loaded data from {file_path}: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
            
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": {},
            "correlations": {},
            "unique_counts": {}
        }
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["summary_stats"] = df[numeric_cols].describe().to_dict()
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                analysis["correlations"] = df[numeric_cols].corr().to_dict()
                
        # Unique value counts for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            analysis["unique_counts"][col] = df[col].value_counts().head(10).to_dict()
            
        return analysis
        
    def create_visualization(self, df: pd.DataFrame, viz_type: str, 
                           columns: List[str] = None, **kwargs) -> str:
        """Create data visualization and return as base64 image"""
        if not PLOTTING_AVAILABLE:
            return "Visualization libraries not available"
            
        try:
            plt.figure(figsize=(10, 6))
            
            if viz_type == "histogram":
                if columns and columns[0] in df.columns:
                    df[columns[0]].hist(bins=30)
                    plt.xlabel(columns[0])
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of {columns[0]}")
                    
            elif viz_type == "scatter":
                if columns and len(columns) >= 2:
                    plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    plt.title(f"{columns[0]} vs {columns[1]}")
                    
            elif viz_type == "line":
                if columns:
                    for col in columns:
                        if col in df.columns:
                            plt.plot(df.index, df[col], label=col)
                    plt.legend()
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.title("Line Plot")
                    
            elif viz_type == "bar":
                if columns and columns[0] in df.columns:
                    value_counts = df[columns[0]].value_counts().head(10)
                    value_counts.plot(kind='bar')
                    plt.xlabel(columns[0])
                    plt.ylabel("Count")
                    plt.title(f"Top 10 {columns[0]} Values")
                    plt.xticks(rotation=45)
                    
            elif viz_type == "correlation":
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
                    plt.title("Correlation Matrix")
                    
            else:
                plt.text(0.5, 0.5, f"Unsupported visualization type: {viz_type}", 
                        ha='center', va='center')
                        
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            plt.close()
            return f"Failed to create visualization: {str(e)}"
            
    def transform_data(self, df: pd.DataFrame, operations: List[Dict]) -> pd.DataFrame:
        """Apply transformations to dataframe"""
        result_df = df.copy()
        
        for op in operations:
            op_type = op.get("type")
            
            if op_type == "filter":
                # Filter rows based on condition
                column = op.get("column")
                condition = op.get("condition")
                value = op.get("value")
                
                if condition == "equals":
                    result_df = result_df[result_df[column] == value]
                elif condition == "greater_than":
                    result_df = result_df[result_df[column] > value]
                elif condition == "less_than":
                    result_df = result_df[result_df[column] < value]
                elif condition == "contains":
                    result_df = result_df[result_df[column].str.contains(value, na=False)]
                    
            elif op_type == "drop_na":
                # Drop rows with missing values
                columns = op.get("columns")
                if columns:
                    result_df = result_df.dropna(subset=columns)
                else:
                    result_df = result_df.dropna()
                    
            elif op_type == "fillna":
                # Fill missing values
                column = op.get("column")
                value = op.get("value", 0)
                if column:
                    result_df[column] = result_df[column].fillna(value)
                else:
                    result_df = result_df.fillna(value)
                    
            elif op_type == "group_by":
                # Group by and aggregate
                group_cols = op.get("columns", [])
                agg_func = op.get("agg_func", "mean")
                if group_cols:
                    result_df = result_df.groupby(group_cols).agg(agg_func).reset_index()
                    
            elif op_type == "sort":
                # Sort values
                column = op.get("column")
                ascending = op.get("ascending", True)
                if column:
                    result_df = result_df.sort_values(by=column, ascending=ascending)
                    
        return result_df
        
    def _handle_load_data(self, query: str) -> str:
        """Handle data loading requests"""
        # Extract file path from query
        # This is simplified - in production would use NLP
        response_parts = ["I'll help you load the data file."]
        
        # Check if file path is mentioned
        if ".csv" in query or ".xlsx" in query or ".json" in query:
            response_parts.append("\nPlease provide the full file path to load the data.")
        else:
            response_parts.append("\nTo load data, please specify:")
            response_parts.append("- File path (e.g., /path/to/data.csv)")
            response_parts.append("- File type (CSV, Excel, JSON)")
            
        return "\n".join(response_parts)
        
    def _handle_analyze_data(self, query: str) -> str:
        """Handle data analysis requests"""
        if not self.dataframes:
            return "No data loaded yet. Please load a dataset first."
            
        # Analyze the most recent dataframe
        df_name = list(self.dataframes.keys())[-1]
        df = self.dataframes[df_name]
        
        analysis = self.analyze_data(df)
        
        response_parts = [
            f"# Data Analysis for {df_name}",
            f"\n## Dataset Overview",
            f"- Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns",
            f"- Columns: {', '.join(analysis['columns'][:10])}{'...' if len(analysis['columns']) > 10 else ''}",
            f"\n## Missing Values"
        ]
        
        missing = analysis['missing_values']
        for col, count in list(missing.items())[:5]:
            if count > 0:
                response_parts.append(f"- {col}: {count} ({count/analysis['shape'][0]*100:.1f}%)")
                
        if analysis['summary_stats']:
            response_parts.append("\n## Summary Statistics")
            for col in list(analysis['summary_stats'].keys())[:3]:
                stats = analysis['summary_stats'][col]
                response_parts.append(f"\n**{col}**:")
                response_parts.append(f"- Mean: {stats.get('mean', 0):.2f}")
                response_parts.append(f"- Std: {stats.get('std', 0):.2f}")
                response_parts.append(f"- Min: {stats.get('min', 0):.2f}")
                response_parts.append(f"- Max: {stats.get('max', 0):.2f}")
                
        return "\n".join(response_parts)
        
    def _handle_visualize_data(self, query: str) -> str:
        """Handle visualization requests"""
        if not self.dataframes:
            return "No data loaded yet. Please load a dataset first."
            
        if not PLOTTING_AVAILABLE:
            return "Visualization libraries are not available. Please install matplotlib and seaborn."
            
        # Determine visualization type
        viz_type = "histogram"  # Default
        if "scatter" in query.lower():
            viz_type = "scatter"
        elif "line" in query.lower():
            viz_type = "line"
        elif "bar" in query.lower():
            viz_type = "bar"
        elif "correlation" in query.lower() or "heatmap" in query.lower():
            viz_type = "correlation"
            
        # Get the most recent dataframe
        df_name = list(self.dataframes.keys())[-1]
        df = self.dataframes[df_name]
        
        # Extract column names from query (simplified)
        columns = []
        for col in df.columns:
            if col.lower() in query.lower():
                columns.append(col)
                
        if not columns and viz_type != "correlation":
            # Use first numeric column for simple plots
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                columns = [numeric_cols[0]]
                if viz_type == "scatter" and len(numeric_cols) > 1:
                    columns.append(numeric_cols[1])
                    
        # Create visualization
        image_data = self.create_visualization(df, viz_type, columns)
        
        if image_data.startswith("data:image"):
            return f"Here's the {viz_type} visualization:\n\n![{viz_type} plot]({image_data})"
        else:
            return image_data
            
    def _handle_transform_data(self, query: str) -> str:
        """Handle data transformation requests"""
        if not self.dataframes:
            return "No data loaded yet. Please load a dataset first."
            
        response_parts = ["I can help you transform your data. Available operations:"]
        response_parts.append("- Filter rows based on conditions")
        response_parts.append("- Drop or fill missing values")
        response_parts.append("- Group by columns and aggregate")
        response_parts.append("- Sort by columns")
        response_parts.append("\nPlease specify what transformation you'd like to perform.")
        
        return "\n".join(response_parts)
        
    def _handle_export_data(self, query: str) -> str:
        """Handle data export requests"""
        if not self.dataframes:
            return "No data loaded yet. Please load and transform a dataset first."
            
        response_parts = ["I can export your data in the following formats:"]
        response_parts.append("- CSV: dataframe.to_csv('output.csv')")
        response_parts.append("- Excel: dataframe.to_excel('output.xlsx')")
        response_parts.append("- JSON: dataframe.to_json('output.json')")
        response_parts.append("\nPlease specify the output format and file name.")
        
        return "\n".join(response_parts)
        
    def _handle_general_data_query(self, query: str) -> str:
        """Handle general data-related queries"""
        capabilities = [
            "# Data Agent Capabilities",
            "\nI can help you with:",
            "- **Load Data**: Import CSV, Excel, JSON files",
            "- **Analyze Data**: Summary statistics, correlations, missing values",
            "- **Visualize Data**: Create charts, plots, and heatmaps",
            "- **Transform Data**: Filter, clean, aggregate, and reshape data",
            "- **Export Data**: Save processed data in various formats",
            "\nWhat would you like to do with your data?"
        ]
        
        return "\n".join(capabilities)