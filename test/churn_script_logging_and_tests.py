'''
Tests of churn library

author: Vittoria Emiliani
date: December 2022
'''
import os
import logging
import sys
import pandas as pd
import joblib

sys.path.append('../')

import churn_library as lib
import constants as c

logging.basicConfig(
    filename='../logs/churn_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


def test_import_data(data_pth):
    '''
    test data import - this example is completed for you to assist with the other test functions

    input:
        data_pth: strinf of data input path
    '''
    try:
        dataframe = lib.import_data(data_pth)
    except FileNotFoundError as err:
        logging.error("Testing import_data: FAILED. The file wasn't found")
        return 'error'

    try:
        assert dataframe.shape[0] > 0, 'dataframe has not rows'
        assert dataframe.shape[1] > 0, 'dataframe has not columns'
    except AssertionError as err:
        logging.error(
            "Testing import_data: FAILED. The file doesn't appear to have rows and/or columns (%s)",
            err)
        return 'error'

    logging.info("Testing import_data: SUCCESS")
    return dataframe


def test_churn_definition(dataframe, attrition_column, value_existing):
    '''
    test churn definition

    input:
        dataframe: pandas dataframe
        attrition_column: column name of attrition column
        value_existing: value to identify existing customers
    '''
    dataframe = lib.churn_definition(dataframe, attrition_column, value_existing)

    try:
        assert sorted(dataframe[c.churn_variable].unique()) == [0, 1]
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing churn definition: FAILED. The 'Churn' is not well defined")
        return err

    return dataframe


def test_perform_eda(dataframe, image_pth):
    '''
    test perform eda function

    input:
        dataframe: pandas dataframe
        image_pth: image output path
    '''
    try:
        lib.perform_eda(dataframe, image_pth)
    except Exception as err:
        logging.error(
            "Testing churn perform_eda: FAILED. some problem with the function")
        return err

    output_files = os.listdir(image_pth)
    try:
        assert 'churn_distribution.png' in output_files, \
            'churn_distribution.png not found'
        assert 'columns_correlation_heatmap.png' in output_files, \
            'columns_correlation_heatmap.png not found'
        assert 'customer_age_distribution.png' in output_files, \
            'customer_age_distribution.png not found'
        assert 'marital_status_distribution.png' in output_files, \
            'marital_status_distribution.png not found'
        assert 'total_trans_ct_distribution.png' in output_files, \
            'total_trans_ct_distribution.png not found'
    except AssertionError as err:
        logging.error(
            "Testing churn perform_eda: FAILED. Not all the "
            "EDA distributions are correctly saved (%s)",
            err)
        return err

    logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper(dataframe: pd.DataFrame, category_lst):
    '''
    test encoder helper

    input:
        dataframe: pandas dataframe
        category_lst: list of columns that contain categorical features
    '''
    dataframe = lib.encoder_helper(dataframe, category_lst)

    try:
        assert 'Gender_Churn' in dataframe.columns, \
            'Gender_Churn is not in columns'
        assert 'Education_Level_Churn' in dataframe.columns, \
            'Education_Level_Churn is not in columns'
        assert 'Marital_Status_Churn' in dataframe.columns, \
            'Marital_Status_Churn is not in columns'
        assert 'Income_Category_Churn' in dataframe.columns, \
            'Income_Category_Churn is not in columns'
        assert 'Card_Category_Churn' in dataframe.columns, \
            'Card_Category_Churn is not in columns'
    except AssertionError as err:
        logging.error(
            'Testing churn encoder_help: FAILED. Some column is not computed (%s)',
            err)
        return err

    try:
        assert dataframe['Gender_Churn'].isnull(
        ).sum() == 0, 'Gender_Churn has null values'
        assert dataframe['Education_Level_Churn'].isnull(
        ).sum() == 0, 'Education_Level_Churn has null values'
        assert dataframe['Marital_Status_Churn'].isnull(
        ).sum() == 0, 'Marital_Status_Churn has null values'
        assert dataframe['Income_Category_Churn'].isnull(
        ).sum() == 0, 'Income_Category_Churn has null values'
        assert dataframe['Card_Category_Churn'].isnull(
        ).sum() == 0, 'Card_Category_Churn has null values'
    except AssertionError as err:
        logging.error(
            'Testing churn encoder_help: FAILED. Some encoding made some nulls values (%s)',
            err)
        return err

    logging.info("Testing encoder_help: SUCCESS")
    return dataframe


def test_perform_feature_engineering(dataframe):
    '''
    test perform_feature_engineering

    input:
              dataframe: pandas dataframe
    '''

    try:
        x_train, x_test, y_train, y_test = lib.perform_feature_engineering(
            dataframe)
    except Exception as err:
        logging.error(
            "Testing features engineering: FAILED. Some error in the execution %s",
            err)
        return err

    try:
        # testing no null output and correct number of row
        assert len(x_train) > 0, 'x_train is empty'
        assert len(x_test) > 0, 'x_test is empty'
        assert len(x_train) == len(
            y_train), 'x_train and y_train have not the same length'
        assert len(x_test) == len(
            y_test), 'y_test and x_test have not the same length'
    except AssertionError as err:
        logging.error(
            "Testing feature engineering: FAILED. Not all the outputs are valorized (%s)",
            err)
        return err

    try:
        x_train_nulls = pd.DataFrame(
            x_train.isnull().sum()).rename(
            columns={
                0: 'nulls'})
        x_test_nulls = pd.DataFrame(
            x_test.isnull().sum()).rename(
            columns={
                0: 'nulls'})

        # testing train & test contains nulls
        assert len(x_train_nulls[x_train_nulls['nulls']
                   > 0]) == 0, 'x_train has null values'
        assert len(x_test_nulls[x_test_nulls['nulls'] > 0]
                   ) == 0, 'x_test has null values'
        assert len(y_train[y_train.isnull()]) == 0, 'y_train has null values'
        assert len(y_test[y_test.isnull()]) == 0, 'y_test has null values'
    except AssertionError as err:
        logging.error(
            "Testing feature engineering: FAILED. Not some feature is null (%s)",
            err)
        return err

    logging.info("Testing feature engineering: SUCCESS")
    return x_train, x_test, y_train, y_test


def test_train_models(
        x_train,
        x_test,
        y_train,
        y_test,
        model_pth,
        result_plot_pth):
    '''
    test train_models

    input:
      x_train: X training data
      x_test: X testing data
      y_train: y training data
      y_test: y testing data
      model_pth: string with the model output path
      result_plot_pth: string with the result plots path
    '''
    try:
        list_output_images = os.listdir(result_plot_pth)
        list_output_models = os.listdir(model_pth)
    except FileNotFoundError as err:
        logging.error(
            "Testing train test model: FAILED. Invalid paths (%s)", err)
        return err

    try:
        lib.train_models(x_train=x_train,
                     x_test=x_test,
                     y_train=y_train,
                     y_test=y_test,
                     model_pth=model_pth,
                     result_plot_pth=result_plot_pth)
    except Exception as err:
        logging.error(
            "Testing train model: FAILED. Some problem with the training")

    try:
        # check if results plots exist
        assert 'logistic_regression_results.png' in list_output_images, \
            'logistic_regression_results.png not present'
        assert 'random_forest_features_importance.png' in list_output_images, \
            'random_forest_features_importance.png not present'
        assert 'random_forest_results.png' in list_output_images, \
            'random_forest_results.png not present'
        assert 'roc_curves_results.png' in list_output_images, \
            'roc_curves_results.png not present'
    except AssertionError as err:
        logging.error(
            "Testing train model: FAILED. Not all the plot results are correctly saved (%s)",
            err)
        return err

    try:
        # check if models object exist
        assert 'logistic_regression.pkl' in list_output_models, \
            'logistic_regression.pkl not present'
        assert 'random_forest.pkl' in list_output_models, \
            'random_forest.pkl not present'

        # check reading
        joblib.load(model_pth + '/logistic_regression.pkl')
        joblib.load(model_pth + '/random_forest.pkl')

    except AssertionError as err:
        logging.error(
            "Testing train model: FAILED. The 2 models was not correctly saved (%s)",
            err)
        return err

    logging.info("Testing train model: SUCCESS")
    return None


if __name__ == "__main__":
    test_import_data(data_pth='./data/bank_data.csv')  # unsuccessful
    df_to_test = test_import_data(
        data_pth='../data/bank_data.csv')  # successful

    test_churn_definition(
        df_to_test.copy(),
        'Attrition_Flag',
        'Existing Customers')  # unsuccessful
    df_to_test = test_churn_definition(
        df_to_test.copy(),
        'Attrition_Flag',
        'Existing Customer')  # successful

    test_perform_eda(df_to_test, image_pth=c.IMAGE_EDA_PATH)  # unsuccessful
    test_perform_eda(
        df_to_test,
        image_pth='../' +
        c.IMAGE_EDA_PATH)  # unsuccessful

    test_encoder_helper(df_to_test,
                        c.cat_columns[:len(c.cat_columns) - 2])  # unsuccessful
    test_encoder_helper(df_to_test, c.cat_columns)  # successful

    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        df_to_test)  # successful

    test_train_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_pth='../' +
        c.MODELS_PATH,
        result_plot_pth='../' +
        c.IMAGE_RESULTS_PATH)  # successful

    test_train_models(x_train=x_train,
                      x_test=x_test,
                      y_train=y_train,
                      y_test=y_test,
                      model_pth=c.MODELS_PATH,
                      result_plot_pth=c.IMAGE_RESULTS_PATH)  # unsuccessful
