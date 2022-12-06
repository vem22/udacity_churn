'''
Library with all the useful functions for Churn project

author: Vittoria Emiliani
date: December 2022
'''

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants as c

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logger = logging.getLogger(name='./log/churn_execution.log')


def import_data(data_pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            data_pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    logger.info("Reading file %s", data_pth)

    try:
        dataframe = pd.read_csv(data_pth)
        return dataframe
    except FileNotFoundError as exception:
        logger.error("Can't find file %s", data_pth)
        raise FileNotFoundError("Can't find file") from exception


def churn_definition(
        dataframe: pd.DataFrame,
        attrition_column: str,
        value_existing: str) -> pd.DataFrame:
    '''
    returns dataframe with churn column

    input:
        dataframe: pandas dataframe
        attrition_column: column name of attrition column
        value_existing: value to identify existing customers

    output:
        dataframe: pandas dataframe with churn column
    '''

    logger.info(
        "definition of churn using column %s with 'customer existing value' = %s",
        attrition_column,
        value_existing)
    dataframe[c.churn_variable] = dataframe[attrition_column].apply(
        lambda val: 0 if val == value_existing else 1)

    return dataframe


def perform_eda(dataframe: pd.DataFrame,
                image_pth: str):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe
            image_pth: image output path

    output:
            None
    '''
    logger.info("Starting EDA")
    logger.info("Df shape %s", dataframe.shape)
    logger.info("Df nulls: %s", dataframe.isnull().sum())
    logger.info("Df distributions: %s", dataframe.describe())

    logger.info("Saving images")
    # distributions
    for column in c.column_distributions:
        plt.figure(figsize=(20, 10))
        plot = sns.histplot(dataframe[column],
                            stat=c.column_distributions[column]['stat'],
                            kde=c.column_distributions[column]['kde'])
        fig = plot.get_figure()

        try:
            output_pth = f'{image_pth}/{column.lower()}_distribution.png'
            fig.savefig(output_pth)
        except Exception as exception:
            logger.error("Cannot save distribution of column %s into %s",
                         column,
                         output_pth)
            logger.error(exception)
            raise Exception(
                'Cannot save distribution of column') from exception

    # correlation heatmap
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    try:
        output_pth = f'{image_pth}/columns_correlation_heatmap.png'
        fig.savefig(output_pth)
    except Exception as exception:
        logger.error("Cannot save heatmap into %s", output_pth)
        logger.error(exception)
        raise Exception('Cannot save heatmap') from exception


def encoder_helper(
        dataframe: pd.DataFrame,
        category_lst: list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            dataframe: pandas dataframe with new columns for
    '''

    # for each categorical variable, make the transformation
    for category_var in category_lst:
        category_groups = dataframe.groupby(category_var).mean()[
            c.churn_variable].to_dict()
        dataframe[category_var + '_' + c.churn_variable] = dataframe[category_var].map(
            lambda x: category_groups.get(x))

    return dataframe


def perform_feature_engineering(dataframe: pd.DataFrame):
    '''
    Perform features transformations and dataset split in train & test sets

    input:
              dataframe: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # encoding category variables
    dataframe = encoder_helper(dataframe=dataframe,
                               category_lst=c.cat_columns)

    # definition of dataframe features and y
    X = dataframe[c.features].copy()
    y = dataframe[c.churn_variable]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def train_models(x_train: pd.DataFrame, x_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series,
                 model_pth: str, result_plot_pth: str):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
              model_pth: string with the model output path
              result_plot_pth: string with the result plots path
    output:
              None
    '''
    # train fandom forest model
    rfc, y_train_preds_rf, y_test_preds_rf = _train_random_forest_model(
        x_train=x_train, x_test=x_test, y_train=y_train)
    _classification_report_image(y_train=y_train,
                                 y_train_preds=y_train_preds_rf,
                                 y_test=y_test,
                                 y_test_preds=y_test_preds_rf,
                                 used_model='Random Forest',
                                 result_plot_pth=result_plot_pth)

    _feature_importance_plots(model=rfc,
                              data=x_test,
                              output_pth=result_plot_pth,
                              used_model='Random Forest')

    save_model(model=rfc,
               output_pth=model_pth,
               output_name='random_forest')

    # train logistic regression model
    lrc, y_train_preds_lr, y_test_preds_lr = _train_logistic_regression_model(
        x_train=x_train, x_test=x_test, y_train=y_train)
    _classification_report_image(y_train=y_train,
                                 y_train_preds=y_train_preds_lr,
                                 y_test=y_test,
                                 y_test_preds=y_test_preds_lr,
                                 used_model='Logistic Regression',
                                 result_plot_pth=result_plot_pth)

    save_model(model=lrc,
               output_pth=model_pth,
               output_name='logistic_regression')

    # roc curve
    _plot_roc_curves(x_test=x_test,
                     y_test=y_test,
                     models={'Random Forest': rfc, 'Logistic Regression': lrc},
                     result_plot_pth=result_plot_pth)


def _train_random_forest_model(x_train, x_test, y_train):
    '''
    train a random forest model
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
    output:
              None
    '''
    logger.info("Training Random Forrest")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=c.param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    y_train_preds = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds = cv_rfc.best_estimator_.predict(x_test)

    return cv_rfc.best_estimator_, y_train_preds, y_test_preds


def _train_logistic_regression_model(x_train, x_test, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
    output:
              None
    '''
    logger.info("Training Logistic Regression")
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    y_train_preds = lrc.predict(x_train)
    y_test_preds = lrc.predict(x_test)

    return lrc, y_train_preds, y_test_preds


def _plot_roc_curves(x_test: pd.DataFrame, y_test: pd.Series, models: dict, result_plot_pth: str):
    '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds: training predictions
                y_test_preds: test predictions
                used_model: string representing the model used (e.g. 'Random forest',
                            'Logistic Regression')
                result_plot_pth: string with the result plots path
        output:
                 None
        '''
    logger.info("Saving ROC curves")
    # plots
    fig = plt.figure(figsize=(15, 8))
    axes = plt.gca()
    for model in models:
        plot_roc_curve(
            models[model], x_test, y_test, ax=axes, alpha=0.8)

    try:
        output_pth = f'{result_plot_pth}/roc_curves_results.png'
        fig.savefig(output_pth)
    except Exception as exception:
        logger.error("Cannot save image %s", output_pth)
        logger.error(exception)
        raise Exception from exception


def _classification_report_image(y_train: pd.Series,
                                 y_test: pd.Series,
                                 y_train_preds: list,
                                 y_test_preds: list,
                                 used_model: str,
                                 result_plot_pth):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions
            y_test_preds: test predictions
            used_model: string representing the model used (e.g. 'Random forest',
                        'Logistic Regression')
            result_plot_pth: string with the result plots path

    output:
             None
    '''
    logger.info("Saving classification report")
    # scores images
    try:
        plt.figure(figsize=(10, 5))
        plt.rc('figure', figsize=(10, 5))
        plt.text(0.01, 1.00, str(f'{used_model} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(classification_report(y_train, y_train_preds)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!

        plt.text(0.01, 0.4, str(f'{used_model} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.005, str(classification_report(y_test, y_test_preds)), {'fontsize': 10},
                 fontproperties='monospace')  # approach improved by OP -> monospace!

        plt.axis('off')
        used_model_output_name = used_model.lower().replace(' ', '_')
        output_pth = f'{result_plot_pth}/{used_model_output_name}_results.png'
        plt.savefig(output_pth)
    except Exception as exception:
        logger.error("Cannot save image %s", output_pth)
        logger.error(exception)
        raise Exception from exception


def _feature_importance_plots(
        model: object,
        data: pd.DataFrame,
        output_pth: str,
        used_model: str):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            data: pandas dataframe of X values
            output_pth: path to store the figure
            used_model: string with the name of the used model
    output:
             None
    '''
    logger.info("Creating features importance")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title(f"Feature Importance {used_model}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(data.shape[1]), names, rotation=90)

    used_model_output_name = used_model.lower().replace(' ', '_')
    try:
        final_output_pth = f'{output_pth}/{used_model_output_name}_features_importance.png'
        plt.savefig(final_output_pth)
    except Exception as exception:
        logger.error("Cannot save image %s", final_output_pth)
        logger.error(exception)
        raise Exception("Cannot save image") from exception



def save_model(model: object, output_pth: str, output_name: str):
    '''
    function to save model object

    input:
        :model: model object to save
        :output_pth: output path where to save
        :output_name: name of the output file

    output:
        None
    '''
    logger.info("Saving model")

    try:
        final_output_pth = f'{output_pth}/{output_name}.pkl'
        joblib.dump(model, final_output_pth)
    except Exception as exception:
        logger.error("Cannot save model %s", final_output_pth)
        logger.error(exception)
        raise Exception("Cannot save model") from exception
