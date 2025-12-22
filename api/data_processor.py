"""
Data processing engine for the Smart Data Analysis Platform
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.processed_data_cache = {}
    
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas DataFrame or None if loading fails
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.csv':
                # Try different encodings and separators
                encodings = ['utf-8', 'latin1', 'iso-8859-1']
                separators = [',', ';', '\t']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                            if len(df.columns) > 1:  # Successful parsing
                                logger.info(f"Successfully loaded CSV with encoding={encoding}, sep='{sep}'")
                                return df
                            uploaded_file.seek(0)  # Reset file pointer
                        except Exception:
                            uploaded_file.seek(0)
                            continue
                
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == '.xlsx' else None)
                return df
            
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return None
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data preprocessing
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # Clean column names
        processed_df.columns = processed_df.columns.str.strip().str.replace(' ', '_')
        
        # Handle datetime columns
        datetime_cols = self.detect_datetime_columns(processed_df)
        for col in datetime_cols:
            try:
                processed_df[col] = pd.to_datetime(processed_df[col])
                logger.info(f"Converted column '{col}' to datetime")
            except Exception as e:
                logger.warning(f"Could not convert '{col}' to datetime: {e}")
        
        # Handle numeric columns
        numeric_cols = self.detect_numeric_columns(processed_df)
        for col in numeric_cols:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                logger.info(f"Converted column '{col}' to numeric")
            except Exception as e:
                logger.warning(f"Could not convert '{col}' to numeric: {e}")
        
        return processed_df
    
    def detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that might contain datetime data
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that might be datetime
        """
        datetime_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column name suggests datetime
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                    datetime_cols.append(col)
                    continue
                
                # Sample some values and try to parse as datetime
                sample_values = df[col].dropna().head(10)
                datetime_count = 0
                
                for value in sample_values:
                    try:
                        pd.to_datetime(str(value))
                        datetime_count += 1
                    except:
                        pass
                
                # If more than 70% of samples can be parsed as datetime
                if datetime_count / len(sample_values) > 0.7:
                    datetime_cols.append(col)
        
        return datetime_cols
    
    def detect_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that might contain numeric data but are stored as strings
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of column names that might be numeric
        """
        numeric_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Sample some values and try to convert to numeric
                sample_values = df[col].dropna().head(20)
                numeric_count = 0
                
                for value in sample_values:
                    try:
                        # Clean the value (remove currency symbols, commas, etc.)
                        cleaned_value = str(value).replace(',', '').replace('$', '').replace('%', '').strip()
                        float(cleaned_value)
                        numeric_count += 1
                    except:
                        pass
                
                # If more than 80% of samples can be converted to numeric
                if len(sample_values) > 0 and numeric_count / len(sample_values) > 0.8:
                    numeric_cols.append(col)
        
        return numeric_cols
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data quality and provide insights
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {}
        
        # Basic statistics
        quality_report['basic_stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        quality_report['missing_data'] = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'total_missing_values': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        quality_report['duplicates'] = {
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        }
        
        # Data type analysis
        dtype_counts = df.dtypes.value_counts().to_dict()
        quality_report['data_types'] = {str(k): v for k, v in dtype_counts.items()}
        
        # Column-wise analysis
        column_analysis = {}
        for col in df.columns:
            col_info = {
                'data_type': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min_value': df[col].min(),
                    'max_value': df[col].max(),
                    'mean_value': df[col].mean(),
                    'std_value': df[col].std()
                })
            
            column_analysis[col] = col_info
        
        quality_report['column_analysis'] = column_analysis
        
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode')
            
        Returns:
            DataFrame with missing values handled
        """
        processed_df = df.copy()
        
        if strategy == 'auto':
            # Automatic strategy based on data characteristics
            for col in processed_df.columns:
                missing_pct = (processed_df[col].isnull().sum() / len(processed_df)) * 100
                
                if missing_pct > 50:
                    # Drop columns with more than 50% missing values
                    processed_df = processed_df.drop(columns=[col])
                    logger.info(f"Dropped column '{col}' due to {missing_pct:.1f}% missing values")
                
                elif processed_df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                
                elif processed_df[col].dtype == 'object':
                    # Fill categorical columns with mode
                    mode_value = processed_df[col].mode()
                    if len(mode_value) > 0:
                        processed_df[col] = processed_df[col].fillna(mode_value[0])
        
        elif strategy == 'drop':
            processed_df = processed_df.dropna()
        
        elif strategy == 'fill_mean':
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
        
        elif strategy == 'fill_median':
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
        
        return processed_df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List]:
        """
        Detect outliers in numeric columns
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            Dictionary with outlier indices for each numeric column
        """
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_outliers = []
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > 3].index.tolist()
            
            if col_outliers:
                outliers[col] = col_outliers
        
        return outliers
