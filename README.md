# Traffic Volume Prediction using Supervised Machine Learning

## Overview
This project predicts traffic volume using supervised machine learning models. The dataset is a time-series traffic dataset, and the goal is to understand traffic patterns and make accurate predictions to help city planners manage traffic efficiently.  

The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization. Both traditional regression models and sequential models (LSTM) are used for prediction.

## Dataset
- **Source:** [Traffic Prediction GRU Starter Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)  
- **Columns Used:**
  - `DateTime` – timestamp of traffic measurement
  - `Vehicles` – number of vehicles (target variable)
  - Additional features derived from DateTime:
    - `Hour`
    - `Day_of_Week`
    - `Weekend` (indicator)

> Note: The dataset CSV should be uploaded manually when running the notebook in Colab.

## Features
- **Data Import & Preprocessing:** Handling missing values, feature engineering (Hour, Day_of_Week, Weekend).  
- **Exploratory Data Analysis (EDA):** Line plots of traffic volume over time, bar plots for average traffic by hour and day of the week.  
- **Model Building:**
  - Linear Regression (baseline)
  - Random Forest Regressor (stronger model)
  - LSTM (advanced sequential model)
- **Evaluation Metrics:** RMSE, R² Score  
- **Visualization:** Actual vs predicted traffic volumes.  

## How to Use
1. Open the notebook in [Google Colab](https://colab.research.google.com).  
2. Upload the dataset CSV when prompted.  
3. Run all cells to reproduce the analysis and predictions.  
4. Optional: Compare the performance of traditional regression models vs LSTM.

## Key Findings
- LSTM significantly outperforms traditional regression models for this dataset.  
- Traffic patterns show strong dependence on **hour of the day** and **day of the week**.  
- Weekends generally have lighter traffic compared to weekdays.  

## Tools & Libraries
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras (for LSTM)  

## How it Helps
City planners can use the model predictions to:  
- Optimize traffic light timings  
- Plan road maintenance or construction schedules  
- Anticipate peak traffic hours and implement preventive measures

## License
This project is for educational purposes and does not have a commercial license.

**Author:** Puja Rani Das  
