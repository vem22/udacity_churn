# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project implements the training of two types of models (Logistic Regression and Random Forrest) that aim to identify credit card customers that are most likely to churn.

## Installation
Before starting running the projects, you should install all the requirements of the project: 
1. Create a new python env, run `conda create -n nameofenv python=3.8 --no-default-packages`
2. Run `pip install -r requirements_py3.8.txt` from the main folder
3. Now your environment is correctly set

## Files and data description
  ```text
     .
    ├── data                   # Input data file 
    ├── images/                # Output images 
    │   ├── eda/               # Storage of Explorative Data Analysis plots (distributions etc.)
    │   └── results/           # Storage of models results plots (features importances etc.)
    ├── logs/                  # Logs files
    ├── models/                # Storage of output models (pickle of models objects)  
    ├── notebooks/             # Notebooks 
    ├── test/                  # Tests 
    ├── churn_main.py          # Main churn file
    ├── churn_library.py       # Library of churn project
    ├── constants.py           # Constants / configuration file
    ├── Guide.ipynb            # Guide of the project
    ├── requirements_py3.8.txt # Python requirements
    └── README.md
  ```

### Data folder
You will find a file named `bank_data.csv` consists of customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc.
This dataset was pulled from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

### Images folder
#### eda
This folder will contain the results of the EDA. Running the project, it will be created those files: 
- `churn_distribution.png`, the dataset churn distribution
- `columns_correlation_heatmap.png`, the correlation heatmap
- `customer_age_distribution.png`, the age distribution
- `marital_status_distribution.png`, the marital status distribution
- `total_trans_ct_distribution.png`, the total transaction count distribution

#### results
This folder will contain the models results plots. Running the project, it will be created those files: 
- `logistic_regression_results.png`, containing the classification report of the logistic regression model; 
- `random_forest_features_importance.png`, containing the features importance of the random forest;
- `random_forest_results.png`, the classification report of the random forest model;
- `roc_curves_results.png`, ROC curves of the two models;

### Logs folder
- `churn_execution.log`, log file of churn execution (churn_main.py);
- `churn_tests.log`, log file of tests execution;

### Models folder
In this folder it will be stored the outputs of the two models. You will find: 
- `logistic_regression.pkl`, object of logistic regression model;
- `random_forest.pkl`, object of random forest model;

### Notebook folder
You will just find `churn_notebook.ipynb`, it is the notebook of the first release of the project

### Test folder
- `churn_script_logging_and_tests.py`, python file that contains unit tests for the churn_library.py functions;

## Running Files
Running the training model will require: 
1. Follow installation instructions; 
2. Activate the environment (run `conda activate envname`)
3. Run `python churn_main.py`

This running will create: 
- EDA images in `images/eda/` folder;
- Model results plots in `images/results/` folder; 
- Model object in `models/` folder; 





