# Credit Risk Prediction Model

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Dataset](#dataset)
- [Features](#features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Streamlit Application](#streamlit-application)
  - [Individual Prediction](#individual-prediction)
  - [Bulk Prediction with CSV Upload](#bulk-prediction-with-csv-upload)
- [Conclusion](#conclusion)


## Project Overview

The **Credit Risk Prediction Model** is a machine learning application designed to predict the likelihood of borrower default on loans. Using features such as borrower income, employment details, and loan characteristics, the model provides an estimate of credit risk, assisting lenders in making informed lending decisions. This project includes both model training and a web application for easy interaction with the model.

The model has been deployed as a **Streamlit** application that offers:
- **Individual Prediction**: Allows users to manually input borrower information to receive a default prediction.
- **Bulk Prediction**: Users can upload a CSV file with multiple borrower records to receive batch predictions, which can be viewed directly and exported for further analysis.

## Problem Statement

In the lending industry, determining the likelihood of a borrower defaulting on a loan is essential to minimize financial losses and manage credit risk effectively. This project addresses the challenge of accurately predicting borrower default risk based on multiple borrower and loan-related features.

## Objective

The objective of this project is to build a predictive model that:
1. Assess the credit risk of individual borrowers based on various features.
2. Provide an app with easy-to-use interface for users to input borrower data and receive predictions.
3. Allow batch processing of credit risk predictions for multiple records via CSV upload.


## Dataset

The model was trained on a dataset containing the following columns:
- `person_age`: Age of the borrower
- `person_income`: Annual income of the borrower
- `person_home_ownership`: Home ownership status of the borrower (e.g., RENT, MORTGAGE, OWN)
- `person_emp_length`: Length of employment in years
- `loan_intent`: Purpose of the loan (e.g., EDUCATION, MEDICAL, PERSONAL)
- `loan_amnt`: Loan amount requested
- `loan_int_rate`: Loan interest rate (as a percentage)
- `loan_percent_income`: Requested Loan amount as percentage of income.
- `cb_person_default_on_file`: Indicates if the borrower has a prior default (Y/N)
- `cb_person_cred_hist_length`: Length of the borrower’s credit history in years

These features are transformed, encoded, and used to train several machine learning models  for binary classification of credit risk out of which Gradient Boosting model turned out to best suited for the purpose of this project.

## Features

- **Prediction of Default Risk**: Predicts the probability of loan default for each borrower.
- **User-Friendly Interface**: Built using Streamlit, allowing easy access for users to input data and view predictions.
- **Batch Predictions**: Supports CSV upload for batch processing of predictions.

## Machine Learning Pipeline

1. **Data Preprocessing**: The dataset undergoes encoding, scaling, and transformation using pre-trained encoders and scalers.

2. **Data Balancing** :To address class imbalance in the dataset, data balancing techniques were applied during model training. The dataset used for credit risk prediction exhibited an uneven distribution between the default and non-default classes. To ensure the model learns effectively from both categories and avoids bias towards the majority class, **oversampling** of the minority class was implemented. This approach improves the model's ability to identify high-risk borrowers accurately, resulting in more reliable predictions in the final application.

4. **Model Training**:Several Machine Learning  models are trained to predict the probability of default based on input features.

5. **Prediction**: The model outputs default probabilities, classifying borrowers as high or low risk based on a set probability threshold.

## Streamlit Application

The application includes two main sections:

### Individual Prediction
The **Credit Risk Prediction** page allows users to input individual borrower details to predict the likelihood of default. Key features of this section include:
- Real-time probability of default prediction.
- Probability breakdown for both default and non-default.

### Bulk Prediction with CSV Upload
The **Bulk Prediction** page allows users to upload a CSV file with multiple borrower records, predicts the default risk for each record, and provides:
- **Data Preview**: Display of the uploaded data with predictions.
- **CSV Export**: Allows users to download the results as a CSV file for further analysis.

**Required CSV Format**:
The uploaded CSV file must contain the following columns in the exact format:
- `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_default_on_file`, `cb_person_cred_hist_length`

## Conclusion

This project demonstrates an effective use of machine learning to assess credit risk, providing a user-friendly application that predicts borrower default risk with high accuracy. By incorporating individual and bulk prediction capabilities, the tool offers financial institutions a streamlined way to analyze risk and make data-driven lending decisions.
