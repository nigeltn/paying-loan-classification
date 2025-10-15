# Customer Payment Status Prediction
This project, developed by "The Byte Squad" for the Data Science Talent Competition 2024 in Vietnam, focuses on predicting whether a customer will be late on their payments (`label=1`) or pay on time (`label=0`). The pipeline involves comprehensive data preprocessing, feature engineering, and a comparative analysis of multiple machine learning models to identify the most accurate predictor.

## Table of Contents
1. [Project Objective](#project-objective)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Development and Comparison](#model-development-and-comparison)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [How to Run](#how-to-run)

## Project Objective
The primary goal is to build a robust classification model that accurately predicts customer payment delinquency. This is a critical task for financial institutions to manage risk and proactively engage with customers. The project addresses common data challenges such as missing values, erroneous data, and significant class imbalance.

## Dataset Description
The dataset contains anonymized features related to a customer's financial history and behavior. The key variables are grouped as follows:
* **Target Variable** (`label`): This is the variable to be predicted.
  * `0`: The customer pays on time.
  * `1`: The customer is late on payments.
* **Feature Groups**:
  * `_COUNT_`: Number of loans categorized by term (short, mid, long) and institution type (bank vs. non-bank).
  * `NUMBER_OF_LOANS_`, `_CREDIT_CARDS_`, `_RELATIONSHIP_`: Total counts of different financial products and relationships.
  * `NUM_NEW_LOAN_TAKEN_xM`: Number of new loans taken by the customer in the last 3, 6, 9, or 12 months.
  * `OUTSTANDING_BAL_`: The outstanding balance on various products (loans, credit cards) at different time intervals (current, 3M, 6M, 9M, 12M ago), including features that represent the change in balance between periods.
  * `INCREASING_BAL_xM`: Features indicating an increase in the outstanding balance over the last 3 or 6 months.
  * `CREDIT_CARD_`: Credit card payment history, such as months since a late payment (10, 30, 60, or 90 days past due) and the total count of late payments.
  * `ENQUIRIES_`: Information related to credit inquiries made by or about the customer.

## Data Preprocessing
The initial dataset contained several issues that required careful handling:
* **Missing Data**: Approximately 10% of the data was missing, affecting 122 out of 124 columns.
* **Erroneous Data**: Some numerical fields, like the number of inquiries (ENQUIRIES), contained invalid negative or decimal values.

To address this, a hybrid approach was used:
1. **Imputation**
   - Logical Correction: For fields like NUMBER OF LOANS, missing totals were calculated by summing bank and non-bank loan counts.
   - Machine Learning Imputation: A Random Forest Regressor was used to predict and fill missing values for `OUTSTANDING LOAN`, `CREDIT CARD MONTH SINCE`, and `NUMBER OF LATE PAYMENT`.
   - Statistical Imputation: Missing values in `INCREASING BAL` were filled with the median.
2. **Data Removal**
   - Rows with insufficient data for reliable imputation (approximately 3% of the dataset) were dropped to maintain data quality without significantly impacting the overall dataset size. 

## Feature Engineering
A systematic, two-stage feature selection process was implemented to identify the most predictive features for the model.
1. **Correlation Analysis**:
   - The `customer_id` column was dropped as it provides no predictive value.
   - An initial selection of the top 40 features was made based on their absolute correlation with the target variable (`label`).
2. **Recursive Feature Elimination (RFE)**:
   - The RFE algorithm, paired with a Random Forest model, was used to iteratively remove the least important features. 
   - This process narrowed the feature set down to the top 20 most influential features, which were used for final model training.

## Model Development and Comparison
The core of the project was to address the significant class imbalance (only 18.76% of samples were label=1) and then train and compare multiple classification models.
* **Handling Imbalance with SMOTE**:
The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data. This technique generates synthetic samples for the minority class to create a balanced dataset, which helps prevent the model from becoming biased towards the majority class.
* **Model Training**: Three models were trained and evaluated to find the best performer:
    - Decision Tree: A baseline model to establish initial performance.
    - Random Forest: An ensemble method to improve stability and accuracy over a single decision tree.
    - XGBoost (Gradient Boosting): An advanced and powerful gradient boosting algorithm for high performance on tabular data.

## Evaluation Metrics
Due to the class imbalance, Accuracy alone is not a sufficient metric. Therefore, the models were evaluated on two key metrics:
* Accuracy Score: The percentage of total correct predictions.
* F1-Score: The harmonic mean of Precision and Recall, providing a better measure of model performance on an imbalanced dataset.

## Results
The performance of the models on the test set is summarized below:

| Model          | Accuracy | F1-Score |
|----------------|----------|----------|
| Decision Tree  | 0.85     | 0.60     |
| Random Forest  | 0.88     | 0.65     |
| XGBoost        | 0.90     | 0.70     |

As shown in the table, the XGBoost model delivered the best performance, demonstrating its effectiveness in handling this classification task.

## How to Run
To replicate this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone
   ```
2. Set up the environment: Ensure you have Python and Conda installed. Create and activate the Conda environment using the provided environment.yml file.
    ```bash
    conda env create -f environment.yml  
    conda activate customer_behavior_prediction
    ```
3. Run the pipeline:
   ```bash
   python -m src.model_compare
   ```