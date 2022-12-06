'''
Constants and configuration file for churn_library and churn_main files

author: Vittoria Emiliani
date: December 2022
'''

# Path Definition
DATA_PATH = "./data/"
IMAGE_EDA_PATH = './images/eda'
IMAGE_RESULTS_PATH = './images/results'
BANK_FILE_PATH = DATA_PATH + "bank_data.csv"
MODELS_PATH = './models'

# Columns Definition
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

churn_variable = 'Churn'

# Distributions configurations
column_distributions = {'Churn': {'stat': 'count', 'kde': False},
                        'Customer_Age': {'stat': 'count', 'kde': False},
                        'Marital_Status': {'stat': 'percent', 'kde': False},
                        'Total_Trans_Ct': {'stat': 'density', 'kde': True}}

# Models configurations
features = ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn']

param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
