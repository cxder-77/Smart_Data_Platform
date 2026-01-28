# 📊  Data Analysis Studio

A powerful **no-code / low-code Streamlit application** for data analysis, visualization, and machine learning. This project allows users to upload datasets, explore them interactively, clean data, generate insights, train ML models, and make predictions — all from a web interface.

---

## 🚀 Features

### 📁 Data Handling

* Upload datasets in **CSV, Excel, and JSON** formats
* Automatic data preview and column analysis
* Memory usage and dataset statistics

### 🔍 Data Profiling & Exploration

* Summary statistics
* Data type inspection
* Missing value analysis
* Distribution plots
* Correlation heatmaps
* Outlier detection

### 📊 Interactive Visualizations

* Bar Charts
* Line Charts
* Scatter Plots
* Histograms
* Box Plots
* Heatmaps
* Pair Plots
* Pie Charts

All charts are interactive using **Plotly**.

### 🧹 Data Cleaning & Transformation

* Handle missing values (mean/median/mode/custom)
* Remove duplicates
* Rename columns
* Change data types
* Encode categorical features
* Normalize / Standardize data
* Create new features
* Filter rows

### 🧠 AI-Powered Insights

* Automatic dataset insights
* Detection of missing values
* Duplicate detection
* Correlation highlights
* Statistical summaries

### 🤖 Machine Learning (No-Code)

Supports both **Regression** and **Classification** problems.

#### Models Included

**Regression:**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

**Classification:**

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

#### ML Capabilities

* Automatic feature encoding
* Smart scaling
* Baseline comparison
* Model evaluation
* Prediction vs Actual plots
* Best model selection
* Manual prediction interface

### 💬 Chat with Data

* Ask simple questions in English
* View statistics, previews, and summaries

### 📤 Export Reports

* Download HTML report
* Export to Excel
* Export to JSON

### 🎨 UI Features

* Dark / Light theme
* Responsive layout
* Modern dashboard interface

---

## 🛠️ Tech Stack

| Technology           | Purpose             |
| -------------------- | ------------------- |
| Python               | Core language       |
| Streamlit            | Web framework       |
| Pandas               | Data processing     |
| NumPy                | Numerical computing |
| Scikit-learn         | Machine Learning    |
| Plotly               | Interactive charts  |
| Matplotlib / Seaborn | Visualization       |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn openpyxl
```

---

## ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📖 How to Use

### Step 1: Upload Data

* Go to **Upload Data**
* Upload CSV / Excel / JSON file

### Step 2: Explore & Clean

* Use **Data Profiling** to understand data
* Use **Data Cleaning** to fix issues

### Step 3: Visualize

* Create charts in **Visualizations** tab
* Use suggested charts

### Step 4: Train ML Model

* Go to **ML Predictions → Configure**
* Select target and features
* Train models

### Step 5: Predict

* Enter values manually
* Click **Predict**
* Get real-time results

### Step 6: Export Report

* Download reports in multiple formats

---

## 📁 Project Structure

```
├── streamlit_app.py      # Main application file
├── requirements.txt     # Dependencies (if available)
├── utils/               # Utility modules (optional)
├── data/                # Sample datasets (optional)
└── README.md            # Project documentation
```

---

## ⚠️ Limitations

* Works best with clean, well-structured datasets
* No persistent database (session-based)
* Advanced NLP chat is limited
* Batch prediction via file upload not yet implemented

---

## 🚧 Future Improvements

* ✅ Batch prediction support
* ✅ Model saving/loading
* ✅ User authentication
* ✅ Cloud deployment
* ✅ Advanced AI chat
* ✅ Database integration
* ✅ Auto feature engineering

---

## 👨‍💻 Author

**Raghav Jha**
BSc IT Student | Data & ML Enthusiast

---

## 📄 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this software.

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub!

For suggestions, issues, or improvements, feel free to open an issue or contact the author.

---

Happy Coding 🚀
