"""
Configuration settings for Smart Data Analysis Platform
"""
import os
from pathlib import Path

# File upload settings
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['.csv', '.xlsx', '.xls']

# Database settings
DATABASE_URL = "sqlite:///./data_platform.db"

# ML Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Chart settings
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_datasets"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_DATA_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# API settings
API_HOST = "localhost"
API_PORT = 8000
