# 🏥 Medical Insurance Cost Prediction

## 📌 Overview
This project is an end-to-end Machine Learning solution to predict individual medical insurance costs based on demographic and health-related factors such as age, BMI, smoking status, and region.

The project covers the complete ML lifecycle including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, experiment tracking using MLflow, and deployment using a Streamlit web application.

---

## 🎯 Problem Statement
Insurance companies need to estimate medical costs for individuals based on their health profile. This project builds a regression model to predict insurance charges accurately.

---

## 📊 Dataset Information
- Total Records: 1337 (after cleaning)
- Features:
  - age
  - sex
  - bmi
  - children
  - smoker
  - region
- Target Variable:
  - `charges` (annual insurance cost)

---

## 🔍 Exploratory Data Analysis (EDA)
Performed:
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Correlation Analysis
- Outlier Detection

### 🔑 Key Insights
- 🚬 Smoking is the most important factor affecting insurance cost
- 📈 Charges increase with age
- ⚖️ High BMI (obesity) leads to higher costs
- 🔥 Obese smokers have the highest charges

---

## ⚙️ Feature Engineering
Created new features to improve model performance:
- BMI Category (Underweight, Normal, Overweight, Obese)
- Age Groups
- Smoker × BMI interaction
- Smoker × Age interaction
- Obese Smoker flag

---

## 🤖 Models Used
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

---

## 📈 Model Performance

| Model               | R² Score |
|--------------------|----------|
| Linear Regression  | **0.9108** 🔥 |
| Lasso Regression   | 0.9108 |
| Ridge Regression   | 0.9107 |
| Gradient Boosting  | 0.8968 |
| XGBoost            | 0.8908 |
| Decision Tree      | 0.8895 |
| Random Forest      | 0.8875 |

### 🏆 Best Model
👉 **Linear Regression** (Highest accuracy with lowest error)

---

## 📊 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## 📈 MLflow Integration
- Tracked all experiments
- Logged parameters and metrics
- Stored models as artifacts
- Registered best model in Model Registry

---

## 🌐 Streamlit Web App
An interactive web application was built using Streamlit.

### Features:
- 📊 EDA Dashboard
- 🤖 Model Performance Comparison
- 🎯 Real-time Prediction
- ⚠️ Risk Level Classification
- 📉 Feature Impact Explanation

### ▶️ Run App:
```bash
streamlit run app.py
```
---
## 📁 Project Structure
```
Medical-Insurance-Cost-Prediction/
│
├── app.py
├── notebooks/
│   └── Medical_Insurance_Cost_Prediction.ipynb
├── data/
│   ├── raw/
│   │   └── medical_insurance.csv
│   └── processed/
│       └── cleaned_medical_insurance.csv
├── images/
│   ├── eda_univariate.png
│   ├── eda_bivariate.png
│   ├── eda_multivariate.png
│   ├── eda_correlation.png
│   ├── eda_outliers.png
│   ├── model_comparison.png
│   └── best_model_analysis.png
├── presentation/
│   └── Medical_Insurance_Prediction.pptx
├── requirements.txt
└── README.md
```
---

## 📦 Installation
```
pip install -r requirements.txt
```
---

## 🧠 Key Learnings
- Importance of feature engineering
- Model comparison and selection
- Experiment tracking using MLflow
- Building end-to-end ML pipeline
- Deploying ML models using Streamlit

---

🎯 Conclusion

This project demonstrates a complete machine learning pipeline from data analysis to deployment. The model performs well with high accuracy and can be used in real-world insurance cost prediction systems.

---

⭐ If you like this project

Give it a ⭐ on GitHub!

---
