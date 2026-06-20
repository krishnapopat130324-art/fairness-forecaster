# 🧠 Fairness Forecaster

### AI-Powered Income Prediction & Bias Detection

An interactive machine learning application that predicts income using the Adult Census dataset while detecting and analyzing bias across race and gender groups.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Fairness Forecaster is more than a traditional machine learning project.

While it accurately predicts whether an individual earns more than $50,000 annually using the Adult Census dataset, it also evaluates the fairness of those predictions by auditing demographic disparities related to race and gender.

The platform combines predictive modeling, explainability, and responsible AI principles into a single interactive dashboard built with Streamlit.

This project demonstrates expertise in:

* Machine Learning
* Data Science
* AI Ethics & Fairness
* Model Explainability
* Software Engineering
* Interactive Web Applications

---

## ✨ Key Features

### 📊 Interactive Data Exploration

Explore feature distributions and analyze relationships between demographic factors and income outcomes through dynamic visualizations.

### 🤖 Machine Learning Model

A Random Forest Classifier trained on the UCI Adult Dataset with approximately 85% prediction accuracy.

### 📈 Comprehensive Evaluation

Detailed performance metrics including:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### 🔍 Feature Importance Analysis

Understand which factors have the greatest impact on income predictions through ranked feature importance visualizations.

### ⚖️ Bias Detection & Fairness Audit

Measure prediction disparities across demographic groups using:

* Positive Prediction Rates
* Group Comparisons
* Disparity Ratios
* Fairness Warnings

### 🔄 Counterfactual Explanations

Generate actionable "what-if" scenarios that reveal the smallest change required to alter a model prediction.

### 💻 Interactive Dashboard

A modern Streamlit-based interface that makes complex fairness metrics easy to understand and explore.

---

## ⚙️ How It Works

### 1️⃣ Data Ingestion & Cleaning

* Load Adult Census Dataset
* Handle missing values
* Encode categorical features
* Prepare data for modeling

### 2️⃣ Exploratory Data Analysis

* Visualize feature distributions
* Analyze target relationships
* Explore correlations

### 3️⃣ Model Training

* Train Random Forest Classifier
* Optimize performance
* Validate predictions

### 4️⃣ Performance Evaluation

Assess model quality using industry-standard metrics.

### 5️⃣ Fairness Analysis

Audit model behavior across demographic groups and calculate disparity ratios.

### 6️⃣ Counterfactual Reasoning

Identify minimal feature changes capable of altering predictions.

---

## 📂 Project Structure

```text
fairness-forecaster/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── adult.data.txt
├── adult.test.txt
└── adult.names.txt
```

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* Git

### Clone Repository

```bash
git clone https://github.com/krishnapopat130324-art/fairness-forecaster.git

cd fairness-forecaster
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 🖥️ Dashboard Sections

### Dataset Overview

* Dataset size
* Class distribution
* Summary statistics

### Exploratory Data Analysis

* Feature distributions
* Income comparisons
* Correlation matrix

### Model Evaluation

* Accuracy metrics
* Confusion matrix
* Classification report
* Feature importance chart

### Fairness Analysis

* Race-based analysis
* Gender-based analysis
* Disparity ratio calculations
* Fairness recommendations

### Counterfactual Analysis

Interactive examples demonstrating how predictions change under different conditions.

---

## 📊 Results

### Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~85%  |
| Precision | High  |
| Recall    | High  |
| F1 Score  | High  |

### Fairness Insights

The system identifies demographic disparities in model outcomes and highlights groups experiencing disproportionate positive or negative prediction rates.

This demonstrates the importance of evaluating fairness alongside predictive performance.

---

## 🌍 Impact

Fairness Forecaster promotes responsible AI by encouraging practitioners to evaluate:

* Accuracy
* Transparency
* Explainability
* Fairness
* Accountability

The project serves as a practical framework for developing trustworthy machine learning systems.

---

## 🛠️ Technologies Used

| Category         | Technology          |
| ---------------- | ------------------- |
| Language         | Python              |
| Data Processing  | Pandas, NumPy       |
| Machine Learning | Scikit-Learn        |
| Visualization    | Matplotlib, Seaborn |
| Web Application  | Streamlit           |
| Version Control  | Git, GitHub         |

---

## 🎯 Skills Demonstrated

### Data Science

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Statistical Analysis

### Machine Learning

* Classification Models
* Random Forests
* Model Evaluation
* Hyperparameter Optimization

### Responsible AI

* Fairness Metrics
* Bias Detection
* Ethical AI Practices

### Explainable AI

* Feature Importance
* Counterfactual Explanations

### Software Engineering

* Modular Design
* Dependency Management
* Documentation
* Version Control

### Web Development

* Interactive Dashboards
* Data Visualization
* User Experience Design

---

## 🔮 Future Enhancements

### Bias Mitigation

* Reweighing
* FairGBM
* Adversarial Debiasing

### Advanced Explainability

* SHAP
* LIME
* Local Explanations

### Real-Time Predictions

* User Input Forms
* Live Inference

### Cloud Deployment

* Streamlit Cloud
* AWS
* Google Cloud Platform
* Azure

---
### Author

* Krishna Popat
