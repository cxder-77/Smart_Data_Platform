import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
import io
import base64
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state first
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'selected_column_dist' not in st.session_state:
    st.session_state.selected_column_dist = None
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = {}

# Theme configuration
THEMES = {
    'light': {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f5f5f5',
        'text_primary': '#000000',
        'text_secondary': '#666666',
        'border': '#e0e0e0',
        'card_bg': '#ffffff',
        'accent1': '#667eea',
        'accent2': '#764ba2',
        'shadow': 'rgba(0,0,0,0.1)'
    },
    'dark': {
        'bg_primary': '#1a1a1a',
        'bg_secondary': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'border': '#404040',
        'card_bg': '#2d2d2d',
        'accent1': '#667eea',
        'accent2': '#764ba2',
        'shadow': 'rgba(0,0,0,0.5)'
    }
}

def get_theme_css():
    """Generate CSS based on current theme"""
    theme = st.session_state.theme
    colors = THEMES[theme]
    
    return f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            --bg-primary: {colors['bg_primary']} !important;
            --bg-secondary: {colors['bg_secondary']} !important;
            --text-primary: {colors['text_primary']} !important;
            --text-secondary: {colors['text_secondary']} !important;
            --border: {colors['border']} !important;
            --card-bg: {colors['card_bg']} !important;
            --accent1: {colors['accent1']} !important;
            --accent2: {colors['accent2']} !important;
            --shadow: {colors['shadow']} !important;
        }}
        
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {{
            background-color: {colors['bg_primary']} !important;
            color: {colors['text_primary']} !important;
        }}
        
        [data-testid="stSidebar"] {{
            background-color: {colors['bg_secondary']} !important;
        }}
        
        [data-testid="stMarkdownContainer"] {{
            color: {colors['text_primary']} !important;
        }}
        
        .main-header {{
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, {colors['accent1']} 0%, {colors['accent2']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, {colors['accent1']} 0%, {colors['accent2']} 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px {colors['shadow']};
        }}
        .insight-card {{
            background: {colors['card_bg']} !important;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {colors['accent1']};
            margin: 10px 0;
            color: {colors['text_primary']} !important;
            box-shadow: 0 2px 4px {colors['shadow']};
        }}
        .stButton>button {{
            width: 100%;
            background: linear-gradient(90deg, {colors['accent1']} 0%, {colors['accent2']} 100%) !important;
            color: white !important;
            border: none;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        [data-testid="stMarkdownContainer"] p {{
            color: {colors['text_primary']} !important;
        }}
        .icon-text {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .icon-text i {{
            font-size: 1.2em;
        }}
        h1, h2, h3, h4, h5, h6, p, span, div {{
            color: {colors['text_primary']} !important;
        }}
    </style>
    """

# Custom CSS with Font Awesome icons
st.markdown(get_theme_css(), unsafe_allow_html=True)# Helper Functions
def load_data(file, file_type):
    """Load data from various file formats"""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type == 'excel':
            df = pd.read_excel(file)
        elif file_type == 'json':
            df = pd.read_json(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_ai_insights(df):
    """Generate AI-powered insights about the dataset"""
    insights = []
    
    # Basic statistics insights
    insights.append(f'<i class="fas fa-chart-bar"></i> Dataset contains {len(df)} rows and {len(df.columns)} columns')
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        insights.append(f'<i class="fas fa-exclamation-triangle"></i> Found {missing.sum()} missing values across {(missing > 0).sum()} columns')
    else:
        insights.append('<i class="fas fa-check-circle"></i> No missing values detected')
    
    # Numeric columns insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            insights.append(f'<i class="fas fa-chart-line"></i> {col}: Mean={mean_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]')
    
    # Categorical insights
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols[:2]:
            unique_count = df[col].nunique()
            top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            insights.append(f'<i class="fas fa-tag"></i> {col}: {unique_count} unique values, Most common: {top_value}')
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        insights.append(f'<i class="fas fa-copy"></i> Found {duplicates} duplicate rows')
    
    # Correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}")
        if high_corr:
            insights.append(f'<i class="fas fa-link"></i> Strong correlations found: {", ".join(high_corr[:3])}')
    
    return insights

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        fig = px.imshow(corr, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Correlation Heatmap")
        return fig
    return None

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def auto_ml_train(df, target_column, problem_type):
    """Automatic ML model training"""
    try:
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target if classification
        if problem_type == 'classification':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {}
        results = {}
        
        if problem_type == 'regression':
            models['Linear Regression'] = LinearRegression()
            models['Decision Tree'] = DecisionTreeRegressor(random_state=42)
            models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                results[name] = {
                    'model': model,
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'predictions': y_pred,
                    'actual': y_test
                }
        else:  # classification
            models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
            models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
            models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'predictions': y_pred,
                    'actual': y_test
                }
        
        return results, scaler, label_encoders, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Error in ML training: {str(e)}")
        return None, None, None, None

def export_to_pdf_html(df, insights, charts):
    """Export analysis as HTML (PDF-ready)"""
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #667eea; }}
            .insight {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #667eea; color: white; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>AI Insights</h2>
        {''.join([f'<div class="insight">{insight}</div>' for insight in insights])}
        
        <h2>Data Preview</h2>
        {df.head(10).to_html()}
        
        <h2>Summary Statistics</h2>
        {df.describe().to_html()}
    </body>
    </html>
    """
    return html_content

def suggest_visualizations(df):
    """Return intelligent, validated visualization suggestions based on data characteristics."""
    suggestions = []
    if df is None or df.empty:
        return suggestions

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()

    # Filter out ID, index, and count columns
    def is_meaningful_col(col_name):
        col_lower = col_name.lower()
        skip_words = ['id', 'index', 'idx', 'count', 'pk', 'rowid', 'serial', 'sequence', 'no.', 'num']
        return not any(word in col_lower for word in skip_words)
    
    meaningful_numeric = [c for c in numeric_cols if is_meaningful_col(c)]
    meaningful_categorical = [c for c in categorical_cols if is_meaningful_col(c)]
    meaningful_low_card_cats = [c for c in meaningful_categorical if 2 <= df[c].nunique() <= 10]
    meaningful_very_low_card_cats = [c for c in meaningful_categorical if df[c].nunique() == 2]
    
    # 1. TIME SERIES - Priority for datetime + numeric
    if datetime_cols and meaningful_numeric:
        suggestions.append({
            "name": "📈 Line Chart (Trend Over Time)",
            "reason": f"Visualize {meaningful_numeric[0]} trends over {datetime_cols[0]}",
        })

    # 2. CATEGORY DISTRIBUTION - Bar chart for meaningful categories
    if meaningful_low_card_cats:
        best_cat = meaningful_low_card_cats[0]
        suggestions.append({
            "name": "📊 Bar Chart (Category Comparison)",
            "reason": f"Compare counts/totals across {best_cat}",
        })

    # 3. NUMERIC DISTRIBUTION - Histogram for continuous data
    if meaningful_numeric:
        suggestions.append({
            "name": "📉 Histogram (Distribution)",
            "reason": f"See the distribution pattern of {meaningful_numeric[0]}",
        })

    # 4. CORRELATION ANALYSIS - Heatmap for numeric relationships
    if len(meaningful_numeric) >= 3:
        suggestions.append({
            "name": "🔥 Heatmap (Correlations)",
            "reason": f"Find relationships between {len(meaningful_numeric[:8])} numeric features",
        })

    # 5. NUMERIC RELATIONSHIP - Scatter for 2+ numeric columns
    if len(meaningful_numeric) >= 2:
        suggestions.append({
            "name": "⚪ Scatter Plot (Relationship)",
            "reason": f"Find patterns between {meaningful_numeric[0]} and {meaningful_numeric[1]}",
        })

    # 6. BINARY CATEGORIES - Pie chart for yes/no or 2-category data
    if meaningful_very_low_card_cats:
        suggestions.append({
            "name": "🥧 Pie Chart (Proportions)",
            "reason": f"Show proportions for {meaningful_very_low_card_cats[0]}",
        })

    # 7. OUTLIER DETECTION - Box plot
    if meaningful_numeric:
        suggestions.append({
            "name": "📦 Box Plot (Outliers & Distribution)",
            "reason": f"Identify outliers and spread in {meaningful_numeric[0]}",
        })

    return suggestions

# Main App
def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<i class="fas fa-database fa-3x" style="color: #667eea;"></i>', unsafe_allow_html=True)
        st.title("Navigation")
        
        menu_items = {
            "Home": "fa-home",
            "Upload Data": "fa-cloud-upload-alt",
            "Data Profiling": "fa-search",
            "Visualizations": "fa-chart-pie",
            "Data Cleaning": "fa-broom",
            "AI Insights": "fa-brain",
            "ML Predictions": "fa-robot",
            "Chat with Data": "fa-comments",
            "Export Report": "fa-file-download"
        }
        
        menu = st.radio("", list(menu_items.keys()), format_func=lambda x: f"{x}")
        
        st.markdown("---")
        
        # Theme toggle
        current_theme_index = 0 if st.session_state.theme == 'Dark' else 1
        theme = st.selectbox("Theme", ["Light", "Dark"], index=current_theme_index, format_func=lambda x: f"🎨 {x}", key="theme_select")
        new_theme = theme.lower()
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Dataset Info")
        if st.session_state.df is not None:
            st.success(f"✓ Loaded: {len(st.session_state.df)} rows")
            st.info(f"Columns: {len(st.session_state.df.columns)}")
            if st.button("Clear Data"):
                st.session_state.df = None
                st.session_state.original_df = None
                st.rerun()
        else:
            st.warning("No data loaded")
    
    # Main content
    if menu == "Home":
        st.markdown('<h1 class="main-header"><i class="fas fa-chart-line"></i> AI Data Analysis Studio</h1>', unsafe_allow_html=True)
        st.markdown("### Transform Your Data into Insights — No Coding Required!")
        
        # Light theme caution
        if st.session_state.theme == 'light':
            st.warning("⚠️ **Under Beta** - Light theme may have display issues with some text and tables. For the best experience, please switch to Dark theme in the sidebar.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('''
            <div class="metric-card">
                <i class="fas fa-cloud-upload-alt fa-3x"></i>
                <p>Upload Data</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown('''
            <div class="metric-card">
                <i class="fas fa-search fa-3x"></i>
                <p>Auto Analysis</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown('''
            <div class="metric-card">
                <i class="fas fa-chart-bar fa-3x"></i>
                <p>Visualize</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            st.markdown('''
            <div class="metric-card">
                <i class="fas fa-brain fa-3x"></i>
                <p>AI Insights</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Key Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - <i class="fas fa-file-upload"></i> **Multi-format Upload**: CSV, Excel, JSON
            - <i class="fas fa-search-plus"></i> **Automated EDA**: One-click profiling
            - <i class="fas fa-chart-area"></i> **Interactive Charts**: Drag & drop visualization
            - <i class="fas fa-tools"></i> **Smart Cleaning**: Auto-handle missing data
            - <i class="fas fa-lightbulb"></i> **AI Insights**: Natural language summaries
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            - <i class="fas fa-robot"></i> **No-Code ML**: Auto-train prediction models
            - <i class="fas fa-comment-dots"></i> **Chat Interface**: Ask questions in plain English
            - <i class="fas fa-file-export"></i> **Export Reports**: PDF, Excel, JSON
            - <i class="fas fa-lock"></i> **Secure**: Session-based, no data retention
            - <i class="fas fa-palette"></i> **Beautiful UI**: Dark/Light themes
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("👈 Start by uploading your data from the sidebar menu!")
    
    elif menu == "Upload Data":
        st.markdown('<h2><i class="fas fa-cloud-upload-alt"></i> Upload Your Dataset</h2>', unsafe_allow_html=True)
        
        upload_type = st.selectbox("Select file type", ["CSV", "Excel", "JSON"])
        uploaded_file = st.file_uploader(
            f"Choose a {upload_type} file",
            type=['csv', 'xlsx', 'xls', 'json']
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file, upload_type.lower())
                if df is not None:
                    st.session_state.df = df.copy()
                    st.session_state.original_df = df.copy()
                    st.success(f"✓ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
                    
                    st.subheader("Data Preview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", len(df))
                    col2.metric("Columns", len(df.columns))
                    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.subheader("Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.values,
                        'Non-Null': df.count().values,
                        'Null': df.isnull().sum().values,
                        'Unique': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
    
    elif menu == "Data Profiling":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first from the 'Upload Data' menu!")
            st.info("← Click on 'Upload Data' in the sidebar to get started")
            return
        
        st.markdown('<h2><i class="fas fa-search"></i> Automated Data Profiling</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
        
        st.markdown("---")
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Sample Values': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Missing Values")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)
            fig = px.bar(missing_data, x='Column', y='Missing %', title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✓ No missing values found!")
        
        st.markdown("---")
        st.subheader("Distribution Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column to analyze", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}', 
                                 marginal='box', nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_col, title=f'Box Plot of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df[selected_col].mean():.2f}")
            col2.metric("Median", f"{df[selected_col].median():.2f}")
            col3.metric("Std Dev", f"{df[selected_col].std():.2f}")
            col4.metric("Skewness", f"{df[selected_col].skew():.2f}")
        
        st.markdown("---")
        st.subheader("Correlation Analysis")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            fig = create_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            corr_matrix = numeric_df.corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False, key=abs)
            
            st.subheader("Top Correlations")
            st.dataframe(corr_df.head(10), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
        
        st.markdown("---")
        st.subheader("Outlier Detection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for outlier detection", numeric_cols, key='outlier_col')
            
            outliers, lower, upper = detect_outliers(df, selected_col)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Outliers", len(outliers))
            col2.metric("Lower Bound", f"{lower:.2f}")
            col3.metric("Upper Bound", f"{upper:.2f}")
            
            if len(outliers) > 0:
                st.warning(f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}% of data)")
                st.dataframe(outliers, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[selected_col], name=selected_col, boxpoints='outliers'))
                fig.update_layout(title=f'Outlier Detection: {selected_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✓ No outliers detected!")
    
    elif menu == "Visualizations":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return

        st.markdown('<h2><i class="fas fa-chart-pie"></i> Interactive Visualizations</h2>', unsafe_allow_html=True)
        df = st.session_state.df

        # Smart suggestion system for first-time users
        st.info("💡 **Pro Tip**: Below are AI-recommended visualizations based on your data. Click 'Use This' to add them!")
        
        suggestions = suggest_visualizations(df)
        
        if suggestions:
            st.subheader("📋 Recommended Visualizations (Choose Below)")
            st.info("Based on your data analysis, here are the best charts for different insights:")
            
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"**{i}. {suggestion['name']}** — {suggestion['reason']}")
        
        st.markdown("---")
        st.subheader("🎨 Create Your Visualization")
        st.info("Select the chart type and columns below to create a custom visualization")
        
        # Initialize visualization configurations list in session
        if "viz_configs" not in st.session_state:
            st.session_state.viz_configs = []

        # Button to add a new visualization block
        if st.button("➕ Add Visualization"):
            st.session_state.viz_configs.append(
                {
                    "type": "Bar Chart",
                    "x": None,
                    "y": None,
                    "color": None,
                    "size": None,
                    "bins": 30,
                    "cols": [],
                }
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        all_cols = df.columns.tolist()

        # Track which visualizations to remove after loop
        remove_indices = []

        for idx, cfg in enumerate(st.session_state.viz_configs):
            st.markdown("---")
            col_title, col_remove = st.columns([5, 1])
            with col_title:
                st.markdown(f"#### Visualization {idx + 1}")
            with col_remove:
                if st.button("🗑️ Remove", key=f"remove_{idx}"):
                    remove_indices.append(idx)
                    st.rerun()

            header_cols = st.columns([3, 1])
            with header_cols[0]:
                viz_type = st.selectbox(
                    "Visualization Type",
                    [
                        "Bar Chart",
                        "Line Chart",
                        "Pie Chart",
                        "Scatter Plot",
                        "Histogram",
                        "Box Plot",
                        "Heatmap",
                        "Pair Plot",
                    ],
                    index=[
                        "Bar Chart",
                        "Line Chart",
                        "Pie Chart",
                        "Scatter Plot",
                        "Histogram",
                        "Box Plot",
                        "Heatmap",
                        "Pair Plot",
                    ].index(cfg["type"]) if cfg["type"] in [
                        "Bar Chart",
                        "Line Chart",
                        "Pie Chart",
                        "Scatter Plot",
                        "Histogram",
                        "Box Plot",
                        "Heatmap",
                        "Pair Plot",
                    ] else 0,
                    key=f"viz_type_{idx}",
                )
            with header_cols[1]:
                if st.button("Remove", key=f"remove_viz_{idx}"):
                    remove_indices.append(idx)
                    continue

            cfg["type"] = viz_type

            # BAR CHART
            if viz_type == "Bar Chart":
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_col = st.selectbox(
                        "X-axis",
                        categorical_cols if categorical_cols else all_cols,
                        key=f"x_{idx}",
                    )
                with c2:
                    y_col = st.selectbox(
                        "Y-axis",
                        numeric_cols if numeric_cols else all_cols,
                        key=f"y_{idx}",
                    )
                with c3:
                    color_col = st.selectbox(
                        "Color by (optional)",
                        [None] + categorical_cols,
                        key=f"color_{idx}",
                    )

                cfg["x"], cfg["y"], cfg["color"] = x_col, y_col, color_col

                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} by {x_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

            # LINE CHART
            elif viz_type == "Line Chart":
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_col = st.selectbox("X-axis", all_cols, key=f"x_{idx}")
                with c2:
                    y_col = st.selectbox("Y-axis", numeric_cols, key=f"y_{idx}")
                with c3:
                    color_col = st.selectbox(
                        "Group by (optional)",
                        [None] + categorical_cols,
                        key=f"color_{idx}",
                    )

                cfg["x"], cfg["y"], cfg["color"] = x_col, y_col, color_col

                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{y_col} over {x_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

            # PIE CHART
            elif viz_type == "Pie Chart":
                c1, c2 = st.columns(2)
                with c1:
                    names_col = st.selectbox(
                        "Categories",
                        categorical_cols if categorical_cols else all_cols,
                        key=f"names_{idx}",
                    )
                with c2:
                    values_col = st.selectbox(
                        "Values",
                        numeric_cols if numeric_cols else all_cols,
                        key=f"values_{idx}",
                    )

                cfg["x"], cfg["y"] = names_col, values_col

                fig = px.pie(
                    df,
                    names=names_col,
                    values=values_col,
                    title=f"Distribution of {values_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

            # SCATTER PLOT
            elif viz_type == "Scatter Plot":
                if len(numeric_cols) < 2:
                    st.warning("Need at least two numeric columns for a scatter plot.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        x_col = st.selectbox(
                            "X-axis", numeric_cols, key=f"x_{idx}"
                        )
                    with c2:
                        y_col = st.selectbox(
                            "Y-axis", numeric_cols, key=f"y_{idx}"
                        )

                    c3, c4 = st.columns(2)
                    with c3:
                        color_col = st.selectbox(
                            "Color by",
                            [None] + categorical_cols + numeric_cols,
                            key=f"color_{idx}",
                        )
                    with c4:
                        size_col = st.selectbox(
                            "Size by",
                            [None] + numeric_cols,
                            key=f"size_{idx}",
                        )

                    cfg["x"], cfg["y"], cfg["color"], cfg["size"] = (
                        x_col,
                        y_col,
                        color_col,
                        size_col,
                    )

                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        size=size_col,
                        title=f"{y_col} vs {x_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # HISTOGRAM
            elif viz_type == "Histogram":
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns available for histogram.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        col = st.selectbox(
                            "Select column", numeric_cols, key=f"hist_{idx}"
                        )
                    with c2:
                        bins = st.slider(
                            "Number of bins",
                            10,
                            100,
                            cfg.get("bins", 30),
                            key=f"bins_{idx}",
                        )

                    cfg["x"], cfg["bins"] = col, bins

                    fig = px.histogram(
                        df,
                        x=col,
                        nbins=bins,
                        title=f"Distribution of {col}",
                        marginal="box",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # BOX PLOT
            elif viz_type == "Box Plot":
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns available for box plot.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        y_col = st.selectbox(
                            "Value column", numeric_cols, key=f"box_y_{idx}"
                        )
                    with c2:
                        x_col = st.selectbox(
                            "Group by (optional)",
                            [None] + categorical_cols,
                            key=f"box_x_{idx}",
                        )

                    cfg["x"], cfg["y"] = x_col, y_col

                    fig = px.box(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"Box Plot of {y_col}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # HEATMAP
            elif viz_type == "Heatmap":
                fig = create_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least two numeric columns for heatmap.")

            # PAIR PLOT
            elif viz_type == "Pair Plot":
                if len(numeric_cols) < 2:
                    st.warning("Need at least two numeric columns for pair plot.")
                else:
                    selected_cols = st.multiselect(
                        "Select columns (max 5)",
                        numeric_cols,
                        default=numeric_cols[: min(3, len(numeric_cols))],
                        key=f"pair_{idx}",
                    )
                    cfg["cols"] = selected_cols

                    if len(selected_cols) >= 2:
                        fig = px.scatter_matrix(
                            df[selected_cols], title="Pair Plot"
                        )
                        fig.update_traces(diagonal_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

        # Remove deleted visualizations
        if remove_indices:
            st.session_state.viz_configs = [
                cfg
                for i, cfg in enumerate(st.session_state.viz_configs)
                if i not in remove_indices
            ]

    
    elif menu == "Data Cleaning":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return
        
        st.markdown('<h2><i class="fas fa-broom"></i> Data Cleaning & Transformation</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Duplicates", "Transform", "Save"])
        
        with tab1:
            st.subheader("Handle Missing Values")
            
            missing_summary = pd.DataFrame({
                'Column': df.columns,
                'Missing': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_summary = missing_summary[missing_summary['Missing'] > 0]
            
            if len(missing_summary) > 0:
                st.dataframe(missing_summary, use_container_width=True)
                
                action = st.selectbox("Select action", [
                    "Fill with mean",
                    "Fill with median",
                    "Fill with mode",
                    "Fill with custom value",
                    "Drop rows with missing values",
                    "Drop columns with missing values"
                ])
                
                if action.startswith("Fill"):
                    columns_to_fill = st.multiselect("Select columns", 
                                                     missing_summary['Column'].tolist())
                    
                    if action == "Fill with custom value":
                        fill_value = st.text_input("Enter fill value")
                    
                    if st.button("Apply"):
                        if action == "Fill with mean":
                            for col in columns_to_fill:
                                if df[col].dtype in [np.float64, np.int64]:
                                    df[col].fillna(df[col].mean(), inplace=True)
                        elif action == "Fill with median":
                            for col in columns_to_fill:
                                if df[col].dtype in [np.float64, np.int64]:
                                    df[col].fillna(df[col].median(), inplace=True)
                        elif action == "Fill with mode":
                            for col in columns_to_fill:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                        elif action == "Fill with custom value":
                            for col in columns_to_fill:
                                df[col].fillna(fill_value, inplace=True)
                        
                        st.session_state.df = df
                        st.success("✓ Missing values handled!")
                        st.rerun()
                
                elif action == "Drop rows with missing values":
                    if st.button("Drop Rows"):
                        df = df.dropna()
                        st.session_state.df = df
                        st.success(f"✓ Dropped rows. New shape: {df.shape}")
                        st.rerun()
                
                elif action == "Drop columns with missing values":
                    cols_to_drop = st.multiselect("Select columns to drop", 
                                                  missing_summary['Column'].tolist())
                    if st.button("Drop Columns"):
                        df = df.drop(columns=cols_to_drop)
                        st.session_state.df = df
                        st.success("✓ Columns dropped!")
                        st.rerun()
            else:
                st.success("✓ No missing values found!")
        
        with tab2:
            st.subheader("Handle Duplicates")
            
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
            
            if duplicates > 0:
                st.warning(f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
                
                if st.button("Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.session_state.df = df
                    st.success(f"✓ Removed duplicates! New shape: {df.shape}")
                    st.rerun()
                
                st.subheader("Preview Duplicates")
                st.dataframe(df[df.duplicated()], use_container_width=True)
            else:
                st.success("✓ No duplicates found!")
        
        with tab3:
            st.subheader("Data Transformation")
            
            transform_type = st.selectbox("Select transformation", [
                "Rename columns",
                "Change data type",
                "Create new column",
                "Normalize/Standardize",
                "Encode categorical",
                "Filter rows"
            ])
            
            if transform_type == "Rename columns":
                col_to_rename = st.selectbox("Select column", df.columns)
                new_name = st.text_input("New name")
                
                if st.button("Rename") and new_name:
                    df = df.rename(columns={col_to_rename: new_name})
                    st.session_state.df = df
                    st.success("✓ Column renamed!")
                    st.rerun()
            
            elif transform_type == "Change data type":
                col = st.selectbox("Select column", df.columns)
                new_type = st.selectbox("New type", ["int", "float", "str", "datetime"])
                
                if st.button("Convert"):
                    try:
                        if new_type == "datetime":
                            df[col] = pd.to_datetime(df[col])
                        else:
                            df[col] = df[col].astype(new_type)
                        st.session_state.df = df
                        st.success("✓ Data type changed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            elif transform_type == "Create new column":
                new_col_name = st.text_input("New column name")
                operation = st.selectbox("Operation", [
                    "Combine columns",
                    "Mathematical operation",
                    "Extract from date"
                ])
                
                if operation == "Combine columns":
                    cols_to_combine = st.multiselect("Select columns", df.columns)
                    separator = st.text_input("Separator", " ")
                    
                    if st.button("Create") and new_col_name and cols_to_combine:
                        df[new_col_name] = df[cols_to_combine].astype(str).agg(separator.join, axis=1)
                        st.session_state.df = df
                        st.success("✓ Column created!")
                        st.rerun()
                
                elif operation == "Mathematical operation":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    col1 = st.selectbox("First column", numeric_cols)
                    operator = st.selectbox("Operator", ["+", "-", "*", "/"])
                    col2 = st.selectbox("Second column", numeric_cols)
                    
                    if st.button("Create") and new_col_name:
                        if operator == "+":
                            df[new_col_name] = df[col1] + df[col2]
                        elif operator == "-":
                            df[new_col_name] = df[col1] - df[col2]
                        elif operator == "*":
                            df[new_col_name] = df[col1] * df[col2]
                        elif operator == "/":
                            df[new_col_name] = df[col1] / df[col2]
                        
                        st.session_state.df = df
                        st.success("✓ Column created!")
                        st.rerun()
            
            elif transform_type == "Normalize/Standardize":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cols_to_scale = st.multiselect("Select columns", numeric_cols)
                method = st.selectbox("Method", ["Standardize (Z-score)", "Normalize (Min-Max)"])
                
                if st.button("Apply") and cols_to_scale:
                    for col in cols_to_scale:
                        if method == "Standardize (Z-score)":
                            df[col] = (df[col] - df[col].mean()) / df[col].std()
                        else:
                            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    
                    st.session_state.df = df
                    st.success("✓ Columns scaled!")
                    st.rerun()
            
            elif transform_type == "Encode categorical":
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                cols_to_encode = st.multiselect("Select columns", categorical_cols)
                
                if st.button("Encode") and cols_to_encode:
                    le = LabelEncoder()
                    for col in cols_to_encode:
                        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    
                    st.session_state.df = df
                    st.success("✓ Columns encoded!")
                    st.rerun()
            
            elif transform_type == "Filter rows":
                col = st.selectbox("Select column", df.columns)
                condition = st.selectbox("Condition", ["equals", "contains", "greater than", "less than"])
                value = st.text_input("Value")
                
                if st.button("Apply Filter") and value:
                    try:
                        if condition == "equals":
                            df = df[df[col] == value]
                        elif condition == "contains":
                            df = df[df[col].astype(str).str.contains(value)]
                        elif condition == "greater than":
                            df = df[df[col] > float(value)]
                        elif condition == "less than":
                            df = df[df[col] < float(value)]
                        
                        st.session_state.df = df
                        st.success(f"✓ Filter applied! Rows remaining: {len(df)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with tab4:
            st.subheader("Save Changes")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Current State"):
                    st.session_state.history.append(df.copy())
                    st.success("✓ State saved!")
            
            with col2:
                if st.button("Reset to Original"):
                    if st.session_state.original_df is not None:
                        st.session_state.df = st.session_state.original_df.copy()
                        st.success("✓ Reset to original data!")
                        st.rerun()
            
            st.subheader("Current Data Preview")
            st.dataframe(df.head(), use_container_width=True)
    
    elif menu == "AI Insights":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return
        
        st.markdown('<h2><i class="fas fa-brain"></i> AI-Powered Insights</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        with st.spinner("Analyzing your data..."):
            insights = generate_ai_insights(df)
        
        st.subheader("Key Insights")
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col1.metric("Total Numeric Cols", len(numeric_cols))
            col2.metric("Avg Correlation", f"{df[numeric_cols].corr().abs().mean().mean():.2f}")
        
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())
        
        st.markdown("---")
        st.subheader("Automated Visualizations")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=numeric_cols[0], title=f'Distribution: {numeric_cols[0]}',
                                 marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_correlation_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    elif menu == "ML Predictions":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return
        
        st.markdown('<h2><i class="fas fa-robot"></i> No-Code Machine Learning</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        st.info("💡 Train predictive models without writing any code!")
        
        tab1, tab2, tab3 = st.tabs(["Configure", "Train & Evaluate", "Predict"])
        
        with tab1:
            st.subheader("Model Configuration")
            
            target_col = st.selectbox("Select target column (what to predict)", df.columns)
            
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                problem_type = 'classification'
                st.info("🎯 Detected: **Classification** problem")
            else:
                problem_type = 'regression'
                st.info("📈 Detected: **Regression** problem")
            
            available_features = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect("Select features (leave empty to use all)", 
                                              available_features,
                                              default=[])
            
            if not selected_features:
                selected_features = available_features
            
            st.write(f"**Selected features ({len(selected_features)}):** {', '.join(selected_features)}")
            
            if st.button("Start Training"):
                st.session_state.target_col = target_col
                st.session_state.problem_type = problem_type
                st.session_state.selected_features = selected_features
                st.success("✓ Configuration saved! Go to 'Train & Evaluate' tab")
        
        with tab2:
            if 'target_col' not in st.session_state:
                st.warning("⚠ Please configure the model first in 'Configure' tab")
            else:
                st.subheader("Train & Evaluate Models")
                
                target_col = st.session_state.target_col
                problem_type = st.session_state.problem_type
                selected_features = st.session_state.selected_features
                
                train_df = df[[target_col] + selected_features].dropna()
                
                st.info(f"Training on {len(train_df)} samples with {len(selected_features)} features")
                
                if st.button("Train Models"):
                    with st.spinner("Training models... This may take a moment..."):
                        results, scaler, encoders, feature_names = auto_ml_train(
                            train_df, target_col, problem_type
                        )
                        
                        if results:
                            st.session_state.ml_results = results
                            st.session_state.ml_scaler = scaler
                            st.session_state.ml_encoders = encoders
                            st.session_state.ml_features = feature_names
                            st.success("✓ Training complete!")
                
                if st.session_state.ml_results:
                    results = st.session_state.ml_results
                    
                    st.subheader("Model Performance")
                    
                    if problem_type == 'regression':
                        comparison_data = []
                        for name, result in results.items():
                            comparison_data.append({
                                'Model': name,
                                'R² Score': f"{result['r2_score']:.4f}",
                                'MAE': f"{result['mae']:.4f}",
                                'MSE': f"{result['mse']:.4f}"
                            })
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
                        st.success(f"🏆 Best Model: **{best_model[0]}** (R² = {best_model[1]['r2_score']:.4f})")
                        
                        st.subheader("Prediction vs Actual")
                        for name, result in results.items():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=result['actual'],
                                y=result['predictions'],
                                mode='markers',
                                name='Predictions',
                                marker=dict(size=8, opacity=0.6)
                            ))
                            fig.add_trace(go.Scatter(
                                x=result['actual'],
                                y=result['actual'],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(
                                title=f'{name} - Predictions vs Actual',
                                xaxis_title='Actual Values',
                                yaxis_title='Predicted Values'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        comparison_data = []
                        for name, result in results.items():
                            comparison_data.append({
                                'Model': name,
                                'Accuracy': f"{result['accuracy']:.4f}",
                                'Precision': f"{result['precision']:.4f}",
                                'Recall': f"{result['recall']:.4f}",
                                'F1 Score': f"{result['f1']:.4f}"
                            })
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                        st.success(f"🏆 Best Model: **{best_model[0]}** (Accuracy = {best_model[1]['accuracy']:.4f})")
        
        with tab3:
            if st.session_state.ml_results is None:
                st.warning("⚠ Please train models first!")
            else:
                st.subheader("Make Predictions")
                st.info("Upload new data or enter values manually to make predictions")
                
                st.markdown("### Manual Input")
                feature_names = st.session_state.ml_features
                input_data = {}
                
                cols = st.columns(3)
                for i, feature in enumerate(feature_names):
                    with cols[i % 3]:
                        input_data[feature] = st.text_input(f"{feature}", key=f"input_{feature}")
                
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        
                        results = st.session_state.ml_results
                        if st.session_state.problem_type == 'regression':
                            best_model_name = max(results.items(), key=lambda x: x[1]['r2_score'])[0]
                        else:
                            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
                        
                        best_model = results[best_model_name]['model']
                        
                        st.success(f"Using: **{best_model_name}**")
                        st.info("🎯 Prediction feature coming soon! (Requires proper encoding of input)")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    elif menu == "Chat with Data":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return
        
        st.markdown('<h2><i class="fas fa-comments"></i> Chat with Your Data</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        st.info("💡 Ask questions about your data in plain English!")
        
        st.markdown("### Example Queries")
        examples = [
            "Show summary statistics",
            "What are the column names?",
            "How many rows and columns?",
            "Show first 10 rows",
            "What is the average of numeric columns?",
            "Show missing values count"
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(examples):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.query = example
        
        query = st.text_input("Ask a question about your data:", 
                             value=st.session_state.get('query', ''))
        
        if st.button("Search") and query:
            with st.spinner("Analyzing..."):
                query_lower = query.lower()
                
                if 'summary' in query_lower or 'statistics' in query_lower:
                    st.subheader("Summary Statistics")
                    st.dataframe(df.describe(), use_container_width=True)
                
                elif 'column' in query_lower and 'name' in query_lower:
                    st.subheader("Column Names")
                    st.write(df.columns.tolist())
                
                elif 'row' in query_lower and 'column' in query_lower:
                    st.subheader("Dataset Shape")
                    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                
                elif 'first' in query_lower or 'head' in query_lower:
                    n = 10
                    st.subheader(f"First {n} Rows")
                    st.dataframe(df.head(n), use_container_width=True)
                
                elif 'average' in query_lower or 'mean' in query_lower:
                    st.subheader("Average of Numeric Columns")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.dataframe(df[numeric_cols].mean(), use_container_width=True)
                
                elif 'missing' in query_lower:
                    st.subheader("Missing Values")
                    missing = df.isnull().sum()
                    missing_df = pd.DataFrame({
                        'Column': missing.index,
                        'Missing Count': missing.values
                    })
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
                
                else:
                    st.info("Advanced natural language queries coming soon! Try one of the example queries above.")
    
    elif menu == "Export Report":
        if st.session_state.df is None:
            st.warning("⚠ Please upload data first!")
            return
        
        st.markdown('<h2><i class="fas fa-file-download"></i> Export Analysis Report</h2>', unsafe_allow_html=True)
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["PDF Report", "Excel Export", "JSON Export"])
        
        with tab1:
            st.subheader("Generate PDF Report")
            
            report_title = st.text_input("Report Title", "Data Analysis Report")
            author = st.text_input("Author Name", "AI Data Studio")
            
            include_options = st.multiselect("Include in report:", [
                "Summary Statistics",
                "Missing Values Analysis",
                "AI Insights",
                "Data Preview"
            ], default=["AI Insights", "Summary Statistics"])
            
            if st.button("Generate HTML Report"):
                with st.spinner("Generating report..."):
                    insights = generate_ai_insights(df)
                    html_report = export_to_pdf_html(df, insights, [])
                    
                    st.download_button(
                        label="Download HTML Report",
                        data=html_report,
                        file_name=f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                    
                    st.success("✓ Report generated! Click above to download")
        
        with tab2:
            st.subheader("Export to Excel")
            
            sheet_name = st.text_input("Sheet Name", "Data")
            include_stats = st.checkbox("Include summary statistics sheet", value=True)
            
            if st.button("Generate Excel"):
                with st.spinner("Generating Excel file..."):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        if include_stats:
                            df.describe().to_excel(writer, sheet_name='Statistics')
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="Download Excel",
                        data=output,
                        file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success("✓ Excel file ready! Click above to download")
        
        with tab3:
            st.subheader("Export to JSON")
            
            orient = st.selectbox("JSON Format", [
                "records",
                "index",
                "columns",
                "values"
            ])
            
            indent = st.checkbox("Pretty print (indented)", value=True)
            
            if st.button("Generate JSON"):
                json_str = df.to_json(orient=orient, indent=4 if indent else None)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                st.success("✓ JSON ready! Click above to download")
                
                st.subheader("Preview")
                st.code(json_str[:500] + "..." if len(json_str) > 500 else json_str, language='json')

if __name__ == "__main__":
    main()
