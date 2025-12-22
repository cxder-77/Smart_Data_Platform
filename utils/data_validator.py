"""
Data validation utilities for the Smart Data Analysis Platform
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from data import config

logger = logging.getLogger(__name__)

class DataValidator:
    """Handles data validation for uploaded files"""
    
    def __init__(self):
        self.max_file_size = config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
        self.allowed_extensions = config.ALLOWED_FILE_TYPES
    
    def validate_file(self, uploaded_file) -> dict:
        """
        Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            dict: Validation result with 'valid' boolean and 'error' message
        """
        try:
            # Check file size
            if uploaded_file.size > self.max_file_size:
                return {
                    'valid': False,
                    'error': (
                        f'File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds '
                        f'maximum allowed size ({config.MAX_FILE_SIZE_MB}MB)'
                    )
                }
            
            # Check file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in self.allowed_extensions:
                return {
                    'valid': False,
                    'error': f'File type "{file_ext}" not supported. '
                             f'Allowed types: {", ".join(self.allowed_extensions)}'
                }
            
            # Try to read the file to ensure it's valid
            try:
                if file_ext == '.csv':
                    pd.read_csv(uploaded_file, nrows=5)
                elif file_ext in ['.xlsx', '.xls']:
                    pd.read_excel(uploaded_file, nrows=5)
                
                # Reset file pointer
                uploaded_file.seek(0)
            
            except Exception as e:
                return {
                    'valid': False,
                    'error': f'File appears to be corrupted or invalid: {str(e)}'
                }
            
            return {'valid': True, 'error': None}
        
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def validate_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Validate DataFrame content.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Validation result with recommendations
        """
        issues = []
        recommendations = []
        
        # Check DataFrame size
        if len(df) == 0:
            issues.append("DataFrame is empty")
        elif len(df) < 10:
            recommendations.append("Dataset is very small - results may not be reliable")
        
        # Check for columns with all missing values
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            issues.append(f"Columns with all missing values: {all_null_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append(f"Found {duplicate_count} duplicate rows - consider removing them")
        
        # Check for columns with single unique value
        single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_cols:
            recommendations.append(
                f"Columns with single unique value (may not be useful): {single_value_cols}"
            )
        
        # Check for object columns that might be numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col])
                    recommendations.append(f"Column '{col}' might be numeric but stored as text")
                except Exception:
                    pass
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_data_pct': (
                    (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    if len(df) > 0 else 0
                ),
                'duplicate_rows': duplicate_count,
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            }
        }
