'''
Main of churn library

author: Vittoria Emiliani
date: December 2022
'''
import logging
import os
import constants as c
import churn_library as lib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(filename='./logs/churn_execution.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

if __name__ == '__main__':
    logging.info('%s Starting churn models process %s', 10 * '*', 10 * '*')

    # import data
    df = lib.import_data(c.BANK_FILE_PATH)

    # target definition
    df = lib.churn_definition(dataframe=df,
                              attrition_column='Attrition_Flag',
                              value_existing='Existing Customer')

    # performing eda
    lib.perform_eda(dataframe=df, image_pth=c.IMAGE_EDA_PATH)

    # data prep
    X_train, X_test, y_train, y_test = lib.perform_feature_engineering(dataframe=df)

    # train models + save results + save outputs
    lib.train_models(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     model_pth=c.MODELS_PATH,
                     result_plot_pth=c.IMAGE_RESULTS_PATH)

    logging.info('%s Process ended %s', 10 * '*', 10 * '*')
