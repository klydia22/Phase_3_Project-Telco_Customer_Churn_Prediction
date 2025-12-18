# Phase_3_Project-Telco_Customer_Churn_Prediction
Data Science Project predicting customer churn using Telco Customer Churn dataset

# **Project Overview**

One of the most significant and persistent challenges faced by telecommunication Companies is Customer churn. It is really expensive  for businesses when a customer leaves, apart from incurring loses in terms of current revenue collection, the business also incurs additional costs for customer replacement.

This project is designed around business impact. The modeling approach prioritizes identifying as many true churners as possible, ensuring that customers who are at risk of leaving are flagged in time for retention.

The project aims at strategizing on how the business will reduce revenue loss by minimizing missed churners through Recall-optimized predictive modeling.

# **Business Problem**

Telecommunication market is highly competitive in that customers can easily switch service providers. While extensive amounts of customer data are collected, many organizations still struggle to convert the collected data into actionable insights that can be meaningful in reducing churn. The project therefore tries to address the lack of reliable data driven strategies that can identify customers who are about to churn. It is certain that when churners are missed; False Negatives, the company loses future revenue, making this type of error more costly than incorrectly flagging a loyal customer. The imbalance in business cost directly informs the modeling strategy adopted in this project.

# **Business Objective**

The primary objective of this project is to offer useful insights that can maximize customer retention. The project will: 

***1. Predict customer churn using historical customer behavior and service usage data***

***2. Maximize the identification of true churners to support proactive retention campaigns***

***3. Maintain model interpretability so results can be trusted and acted upon by stakeholders***

***4. Translate model outputs into clear, business-oriented recommendations***

# ***Import Libraries***
import pandas as pd
import numpy as np

# For modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt

# **Dataset Description**

The analysis uses Telco Customer Churn dataset that contains detailed records of customer demographics, subscribed services, billing information and contract details. The dataset has a record of 7,043 customers and 21 features. The features include; gender, senior citizen status, tenure, contract type, payment method, internet service, streaming services, tech support, monthly charges, total charges and the target variable 'Churn' (Yes/No).

The dataset provides a realistic representation of customer behavior in the telecom industry and is well-suited for churn prediction activities.

import pandas as pd
import numpy as np

df = pd.read_csv('Telco_Customer_Churn.csv')
print(df.info())

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB

# **Data Preparation**

Raw data most of the time contains inconstistencies and formatting issues that can negatively impact the performance of the model. This necessesitated the need for the data to undergo several preprocessing steps to ensure data quality and reliability. The following were undertaken:

***converted TotalCharges from a string format to numeric values to enable proper mathematical operations***

***Handled blank strings and missing values to prevent downstream modeling errors***

***Droped the customerID colum, as unique identifiers do not contribute predictive value***

***Transformed the target variable into a binary format suitable for classification models***

The steps above ensured that the dataset was both clean and analytically sound before modeling.

# Introduce missing values in TotalCharges, replicating the real dataset issue
df['TotalCharges'] = df['TotalCharges'].astype(str)
for i in np.random.choice(df.index, 11,replace=False) : # 11 is the usual number of missing values
    df.loc[i, 'TotalCharges'] = ' '
    
# columns with empty strings
# Convert TotalCharges to numeric, turning spaces into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill the new NaNs 
df['TotalCharges']= df['TotalCharges'].fillna(0)

# Converting target variable to Binary (0 or 1)

df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x== 'Yes' else 0)

# **Feature Engineering & Preprocessing**

The dataset contained a mixture of numeric and categorical variables which necessitated the implementation of a structured preprocessing pipeline to ensure consistency and prevent data leakage.

Numeric features such as tenure and charges were imputed using the median to reduce the influence of outliers and scaled using StandardScaler to ensure equal contribution to model training.

Categorical variables were transformed using One-Hot Encoding and handled carefully to account for unseen categories during testing.

***Pipeline Design***

A columnTransformer and Pipeline architecture was used so that all preprocessing steps were learned exclusively from the training data and  applied automatically to new, unseen data. This ensures reproducibility, scalability and protection against data leakage. 

# **Data Modeling**

The project explored two complementary modeling techniques to balance interpretability and predictive power.

# ***Logistic Regression***

This model served as the baseline model due to its transparency and ease of interpretation. Model coefficients provide clear insight into how each feature influences the likelihood of churn, making this model valuable for business communication.

# ***Decision Tree Classifier***

This model was introduced to capture non-linear relationships and interactions between features that Logistic Regression may miss. Decision Trees also provide intuitive feature importance metrics that align well with stakeholder expectations.

Both models were tuned using GridSearchCV with Recall as the optimization objective.
# **Handling Imbalance**

The dataset exhibits a natural class imbalance with substantially more non-churners than churners. Without corrective measures, models tend to favor the majority class and underperform in identifying churrners. This issue was addressed by applying SMOTE (Synthetic Minority Oversampling Technique), oversampling was restricted to the training dataset to avoid information leakage, SMOTE was embedded directly within the modeling pipeline.

The strategy greatly improved the model's sensitivity to churn behavior.

# **Model Performance**

Model performance was evaluated primarily using Recall, reflecting the business priority of catching as many churners as possible.

# **Model Recall**

Baseline Logistic Regression	~55%
Tuned Logistic Regression + SMOTE	~79%
Tuned Decision Tree + SMOTE	~75%

The tuned Logistic Regression model with SMOTE delivered the strongest Recall, making it the preferred solution for churn intervention.
# **Key Insights**

The analysis of model coefficients and feature importance revealed several consistent churn drivers as follows:

***Customers on month-to-month contracts are significantly more likely to churn than those on long-term contracts***

***Low tenure customers exhibit higher churn risk, highlighting the importance of early engagement***

***Fiber optic internet service is associated with elevated churn, suggesting potential service or pricing issues***

***Long-term contracts serve as strong retention mechanisms***

The insights above provide actionable levers for customer retention strategies.
# **Business Impact**

By shifting the modeling focus towards Recall, this project delivers tangible business value:

***Improves early identification of at-risk customers***

***Enables targeted and cost-effective retention campaigns***

***Reduces long-term revenue loss associated with customer churn***

***Bridges the gap between data science outputs and business decision-making***

# **Recommendations**

Based on the model findings, the following actions are recommended:

***1. Prioritize retention offers for customers on month-to-month contracts***

***2. Implement onboarding and engagement programs for new and low-tenure customers***

***3. Conduct deep analysis into fiber optic service dissatisfaction***

***4. Integrate churn prediction outputs into CRM systems for real-time intervention***

# **Limitations**

Logistic regression assumes linear relationships between features and churn. SMOTE introduces synthetic data that may not fully capture real-world behavior.

The model does not yet account for customer lifetime value or intervention costs.

