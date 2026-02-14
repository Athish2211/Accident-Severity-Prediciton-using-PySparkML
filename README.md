# 🚗 Motor Vehicle Collisions Analysis - Big Data Analytics

A comprehensive big data analytics project using **PySpark** to analyze motor vehicle collision data, perform exploratory data analysis (EDA), and build machine learning models to predict crash severity.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

## 🔍 Overview

This project analyzes motor vehicle collision data using Apache Spark (PySpark) to handle large-scale data processing. The analysis includes comprehensive exploratory data analysis (EDA), feature engineering, and the implementation of multiple machine learning models to predict crash severity levels.

**Key Objectives:**
- Perform scalable big data analysis on vehicle collision records
- Identify patterns and trends in crash data across different boroughs
- Build predictive models to classify crash severity (No Injury, Injury, Death)
- Compare performance of Logistic Regression vs XGBoost models

## 📊 Dataset

**Source**: Motor Vehicle Collisions - Crashes dataset

**Features:**
- Location data (Borough, Latitude, Longitude)
- Temporal data (Crash Date, Crash Time)
- Injury/fatality counts (Persons, Pedestrians, Cyclists, Motorists)
- Contributing factors and vehicle types
- Multiple categorical and numerical features

**Target Variable**: SEVERITY
- **0**: No injuries or deaths
- **1**: Injuries occurred
- **2**: Deaths occurred

## ✨ Features

### Data Processing
- ✅ **Distributed Computing**: Leverages PySpark for scalable data processing
- ✅ **Missing Value Handling**: Comprehensive null value treatment
- ✅ **Feature Engineering**: Temporal features extraction (hour, day of week, month)
- ✅ **Data Transformation**: String indexing for categorical variables
- ✅ **Pipeline Architecture**: Automated ML pipelines for reproducibility

### Analysis Capabilities
- 📈 **Statistical Analysis**: Descriptive statistics and distribution analysis
- 🔍 **Correlation Analysis**: Feature correlation with severity
- 📊 **Visualization**: Matplotlib and Seaborn visualizations
- 🎯 **Model Comparison**: Multiple algorithms evaluation

### Machine Learning
- 🤖 **Logistic Regression**: Baseline classification model
- 🚀 **XGBoost**: Advanced gradient boosting classifier
- 📏 **Performance Metrics**: Accuracy, F1-Score, Precision, Recall
- 📉 **Confusion Matrix**: Detailed prediction analysis

## 🛠️ Technologies Used

### Big Data & ML
- **Apache Spark (PySpark)**: Distributed data processing
- **PySpark SQL**: Data manipulation and transformation
- **PySpark ML**: Machine learning pipelines
- **XGBoost**: Gradient boosting framework

### Data Analysis & Visualization
- **Pandas**: Data manipulation for visualizations
- **NumPy**: Numerical computations
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

### Environment
- **Jupyter Notebook**: Interactive development
- **Python 3.x**: Programming language

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Apache Spark 3.x
- Java 8 or 11 (required for Spark)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/motor-vehicle-collisions-analysis.git
cd motor-vehicle-collisions-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create `requirements.txt`**:
```txt
pyspark>=3.2.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
```

4. **Download the dataset**:
Place `Motor_Vehicle_Collisions_-_Crashes.csv` in the appropriate directory.

5. **Run Jupyter Notebook**:
```bash
jupyter notebook BDA_AAT_NoteBook__1_.ipynb
```

## 📈 Project Workflow

```
1. Data Loading
   ├── Initialize Spark Session
   └── Load CSV into PySpark DataFrame

2. Data Inspection
   ├── Schema Analysis
   ├── Sample Data Review
   └── Summary Statistics

3. Data Cleaning
   ├── Missing Value Analysis
   ├── Null Value Handling
   └── Data Type Corrections

4. Feature Engineering
   ├── Temporal Features (Hour, Day, Month)
   ├── Severity Classification
   └── Categorical Encoding

5. Exploratory Data Analysis
   ├── Distribution Analysis
   ├── Correlation Analysis
   └── Visualization

6. Model Building
   ├── Data Preparation
   ├── Feature Assembly
   ├── Train/Test Split (80/20)
   └── Pipeline Creation

7. Model Training & Evaluation
   ├── Logistic Regression
   ├── XGBoost Classifier
   └── Performance Comparison

8. Results & Insights
   ├── Confusion Matrix
   ├── Metrics Evaluation
   └── Model Comparison
```

## 🔬 Exploratory Data Analysis

### Key Analyses Performed

**1. Distribution Analysis**
- Crashes by Borough (line plot)
- Number of persons injured (histogram)
- Temporal patterns (hourly, daily, monthly)

**2. Missing Value Analysis**
```python
# Missing values identified and counted for each column
# Strategic imputation:
# - Categorical: 'Unknown'
# - Numerical: 0
```

**3. Correlation Matrix**
- Identifies relationships between numerical features
- Highlights predictors of severity
- Heatmap visualization with Seaborn

**4. Feature Engineering**
```python
# Temporal Features
- CRASH_HOUR: Extracted from crash time
- CRASH_DAYOFWEEK: Day of the week (1-7)
- CRASH_MONTH: Month of the year (1-12)

# Severity Classification
SEVERITY = {
    2: if deaths > 0
    1: if injuries > 0
    0: otherwise
}
```

## 🤖 Machine Learning Models

### Model 1: Logistic Regression

**Configuration:**
```python
LogisticRegression(
    featuresCol="features",
    labelCol="SEVERITY",
    maxIter=50,
    regParam=0.01
)
```

**Features Used:**
- **Numerical**: Latitude, Longitude, Crash Hour, Day of Week, Month
- **Categorical (Indexed)**: Borough, Contributing Factor, Vehicle Types

### Model 2: XGBoost Classifier

**Configuration:**
```python
SparkXGBClassifier(
    features_col="features",
    label_col="SEVERITY",
    numClass=3,
    maxDepth=8,
    eta=0.1,
    numRound=150,
    subsample=0.8,
    colsampleByTree=0.8
)
```

**Advantages:**
- Handles imbalanced data better
- Captures non-linear relationships
- Built-in regularization

### Pipeline Architecture

```python
Pipeline(stages=[
    # Stage 1: String Indexing
    StringIndexer for categorical features,
    
    # Stage 2: Feature Assembly
    VectorAssembler,
    
    # Stage 3: Model Training
    Classifier (LR or XGBoost)
])
```

## 📊 Results

### Evaluation Metrics

Both models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Harmonic mean of precision and recall
- **Weighted Precision**: Precision across all classes
- **Weighted Recall**: Recall across all classes

### Confusion Matrix Analysis

```python
# Confusion matrix shows:
# - True Positives/Negatives
# - False Positives/Negatives
# - Per-class performance
```

### Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | XX.XX% | X.XXX | X.XXX | X.XXX |
| XGBoost | XX.XX% | X.XXX | X.XXX | X.XXX |

*Note: Run the notebook to see actual metrics*

## 📁 Project Structure

```
motor-vehicle-collisions-analysis/
│
├── BDA_AAT_NoteBook__1_.ipynb      # Main analysis notebook
├── Motor_Vehicle_Collisions.csv    # Dataset (not included in repo)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Saved models (generated)
│   ├── logistic_regression_model/
│   └── xgboost_model/
│
└── visualizations/                 # Generated plots (optional)
    ├── correlation_heatmap.png
    ├── borough_distribution.png
    └── confusion_matrices.png
```

## 💻 Usage

### Running the Complete Analysis

```python
# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Motor Vehicle Collisions EDA") \
    .getOrCreate()

# 2. Load Data
df = spark.read.csv('Motor_Vehicle_Collisions_-_Crashes.csv', 
                    header=True, 
                    inferSchema=True)

# 3. Run EDA
df.printSchema()
df.describe().show()

# 4. Build and Train Models
# (Follow notebook cells sequentially)

# 5. Evaluate Results
evaluator.evaluate(predictions)
```

### Making Predictions on New Data

```python
# Load trained model
model = PipelineModel.load("path/to/saved/model")

# Prepare new data
new_data = spark.createDataFrame([...])

# Make predictions
predictions = model.transform(new_data)
predictions.select("SEVERITY", "prediction", "probability").show()
```

## 🔮 Future Enhancements

### Data & Features
- [ ] **Time Series Analysis**: Trend analysis over years
- [ ] **Weather Integration**: Include weather conditions
- [ ] **Traffic Data**: Incorporate traffic volume data
- [ ] **Geospatial Analysis**: Hot-spot identification using clustering

### Models
- [ ] **Random Forest**: Ensemble method implementation
- [ ] **Neural Networks**: Deep learning with Spark DL
- [ ] **Hyperparameter Tuning**: Grid search optimization
- [ ] **Class Imbalance**: SMOTE or weighted classes

### Deployment
- [ ] **Real-time Prediction**: Streaming data with Spark Streaming
- [ ] **Dashboard**: Interactive visualization with Dash/Streamlit
- [ ] **API Endpoint**: REST API for predictions
- [ ] **Model Monitoring**: MLflow integration

## 📈 Key Insights

From the analysis, we can derive:

1. **Temporal Patterns**: Certain hours and days show higher collision rates
2. **Geographic Distribution**: Borough-wise accident concentration
3. **Severity Predictors**: Key factors contributing to severe crashes
4. **Model Performance**: Comparison of traditional vs gradient boosting approaches

## 🎓 Learning Outcomes

This project demonstrates:
- **Big Data Processing**: Handling large datasets with PySpark
- **Distributed Computing**: Leveraging Spark's distributed architecture
- **ML Pipelines**: Building reproducible ML workflows
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Comparison**: Evaluating multiple algorithms
- **Data Visualization**: Communicating insights effectively

## 👏 Acknowledgments

- NYC Open Data for the Motor Vehicle Collisions dataset
- Apache Spark community
- XGBoost developers
- Contributors and data science community

---

**Made with ⚡ PySpark and 📊 Data Science**

*For educational and analytical purposes. Analysis based on publicly available data.*
