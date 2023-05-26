import os.path
import datetime as dt
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
import app.config as cfg
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from app.data.data_management import create_directory, save_dict_as_pickle, load_dict_from_pickle
from app.data.data_utils import DATA_LOADER, get_level_wise_contribution

print(sys.path)

import warnings

warnings.filterwarnings('ignore')

'''
Utility Function to Create Name Of Day column for a particular Date.
'''


def create_day_name_col(df, date_col=None):
    df[cfg.DAY_NAME_COL] = df[date_col].dt.day_name()
    return df


def check_for_particular_day(df, day_name_col=None, day_name_list=None):
    var_name = '_or_'.join(day_name_list)

    df[var_name] = df[day_name_col].apply(lambda x: abs(int(x in day_name_list) - 0.3))

    return df


def check_for_days(day, day_name_list=None):
    check = day in day_name_list

    return abs(int(check) - 0.3)


'''
TRAINING UTILITY FUNCTIONS

Function to run grid search on SARIMAX and return the optimal parameters
'''


def sarimax_grid_search(df, exogenous_col=None, target_col=None, seasonal_period=None):
    if exogenous_col:
        result = auto_arima(df[target_col], X=df[[exogenous_col]], start_p=1, start_q=1, max_p=7,
                            max_q=7, max_d=7,
                            m=seasonal_period, seasonal=True, step_wise=True)
    else:

        result = auto_arima(df[target_col], start_p=1, start_q=1, max_p=7,
                            max_q=7, max_d=7,
                            m=seasonal_period, seasonal=True, step_wise=True)

    return result.get_params()


def fit_sarimax_model(data, exog_data=None, params_dict=None):
    model = SARIMAX(data, exog=exog_data, **params_dict, enforce_invertibility=False, enforce_stationarity=False).fit(
        disp=False)

    return model


def build_sarimax_model(df, target_col=None, city_col=None, city_value=None, l3_col=None,
                        l3_value=None, n_days=None, seasonal_period=None, date_col=None, exog_col=None):
    df_temp = df[df[city_col] == city_value]
    if l3_value:
        df_temp = df_temp[df_temp[l3_col] == l3_value]

    df_temp = df_temp.iloc[-n_days:]
    df_temp.set_index(date_col, inplace=True)
    df_temp.index.freq = 'D'

    result = sarimax_grid_search(df_temp, exogenous_col=exog_col, target_col=target_col,
                                 seasonal_period=seasonal_period)

    if exog_col:
        exog_data = df_temp[[exog_col]]
    else:
        exog_data = None

    model = fit_sarimax_model(df_temp[target_col], exog_data=exog_data, params_dict=result)

    return model


def train_l3_wise_sarimax_and_save(df, target_col=None, city_col=None, city_value=None, l3_col=None, n_days=None,
                                   seasonal_period
                                   =None, date_col=None, exog_col=None, model_file_path=None):
    model_dict = {}
    df_city = df[df[city_col] == city_value].copy()

    for l3 in df_city[l3_col].unique():
        df_l3 = df_city[df_city[l3_col] == l3]

        m = build_sarimax_model(df_l3, target_col=target_col, city_col=city_col, city_value=city_value, l3_col=l3_col,
                                l3_value=l3, n_days=n_days, seasonal_period=seasonal_period, date_col=date_col,
                                exog_col=exog_col)

        key = f'{city_value}_{l3}'
        value = m

        model_dict[key] = value

    #     print(f'models for {city_value} built')

    path = create_directory(root_path=model_file_path, directory_name=city_value)
    save_dict_as_pickle(model_dict, file_name=f'{city_value}.pkl', dir_name=path)

    return


def train_city_l3_wise_sarimax_and_save(df, city_col=None, target_col=None, l3_col=None, n_days=None,
                                        seasonal_period=None, date_col=None, exog_col=None, model_file_path=None,
                                        error_file_path=None):
    error_city_list = []
    error_list = []

    for city in df[city_col].unique():

        try:
            train_l3_wise_sarimax_and_save(df, target_col=target_col, city_col=city_col, city_value=city, l3_col=l3_col,
                                           n_days=n_days, seasonal_period
                                           =seasonal_period, date_col=date_col, exog_col=exog_col,
                                           model_file_path=model_file_path)

        except Exception as e:

            error_city_list.append(city)
            error_list.append(str(e))

        error_df = pd.DataFrame({'city': error_city_list, 'error_msg': error_list})

        error_path = os.path.join(error_file_path, 'training_errors.csv')
        error_df.to_csv(error_path, index=False)

    return None


'''
PREDICTION UTILITY FUNCTIONS
'''


# def generate_city_wise_top_categories(city_col=None, model_file_path=None, city_value=None, start_date=None, steps=None,
#                                       max_model_date=None,
#                                       day_name_list=None, top_n_results=None):
#     '''
#     city_col and max_model_date must be passed
#     pass max_model_date as string always
#     pass start_date as string
#     '''
#
#     column_names = ['Date', 'City', 'L3_Category', 'Forecasted_Value']
#     result_df = pd.DataFrame(columns=column_names)
#
#     if not start_date:
#         start_date = pd.to_datetime(dt.datetime.today().date())
#
#     else:
#         start_date = pd.to_datetime(start_date)
#
#     max_model_date = pd.to_datetime(max_model_date) + dt.timedelta(days=1)
#
#     gap_days = (start_date - max_model_date).days
#
#     steps += gap_days
#
#     end_date = max_model_date + dt.timedelta(days=steps - 1)
#
#     days = [_.day_name() for _ in pd.date_range(max_model_date, end_date)]
#
#     exog_array = [[check_for_days(day, day_name_list=day_name_list) for day in days]]
#
#     path = os.path.join(model_file_path, city_value)
#
#     model_dict = load_dict_from_pickle(file_name=f'{city_value}.pkl', dir_name=path)
#
#     for key, model in model_dict.items():
#         city = key.split('_')[0]
#         l3 = key.split('_')[1]
#
#         forecast = model.forecast(steps=steps, exog=exog_array, typ='levels')
#         date = str(forecast.index.date[-1])
#
#         result_df.loc[len(result_df.index)] = [date, city, l3, np.round(forecast[-1],2)]
#
#         result_df.sort_values(by='Forecasted_Value', inplace=True, ascending=False, ignore_index=True)
#
#         result_df = result_df[result_df['Forecasted_Value']!= 0]
#
#     return result_df.iloc[:top_n_results][['L3_Category', 'Forecasted_Value']].set_index('L3_Category')['Forecasted_Value'].to_dict()


## Tested sucecessfully: generate_city_wise_top_categories not working for ADONI because quantity model not present. Rest is working fine.
def generate_city_wise_top_categories(model_file_path=None, city_value=None, start_date=None, steps=None,
                                      max_model_date=None, day_name_list=None, top_n_results=None, model_types=None,
                                      weights=None, bottom=None):
    '''
    city_value and max_model_date must be passed
    pass max_model_date as string always
    pass start_date as string
    '''

    final_df = pd.DataFrame()

    if not start_date:
        start_date = pd.to_datetime(dt.datetime.today().date())

    else:
        start_date = pd.to_datetime(start_date)

    max_model_date = pd.to_datetime(max_model_date) + dt.timedelta(days=1)

    gap_days = (start_date - max_model_date).days

    steps += gap_days

    end_date = max_model_date + dt.timedelta(days=steps - 1)

    days = [_.day_name() for _ in pd.date_range(max_model_date, end_date)]

    exog_array = [[check_for_days(day, day_name_list=day_name_list) for day in days]]

    for model_type in model_types:

        column_names = ['Date', 'City', 'L3_Category', f'{model_types[model_type]}_forecast']
        result_df = pd.DataFrame(columns=column_names)

        new_model_file_path = os.path.join(model_file_path, model_types[model_type])

        new_model_file_path = os.path.join(new_model_file_path, city_value)

        print(new_model_file_path)

        model_dict = load_dict_from_pickle(file_name=f'{city_value}.pkl', dir_name=new_model_file_path)
        # print(model_dict)

        try:
            for key, model in model_dict.items():
                city = key.split('_')[0]
                l3 = key.split('_')[1]

                forecast = model.forecast(steps=steps, exog=exog_array, typ='levels')
                date = str(forecast.index.date[-1])

                result_df.loc[len(result_df.index)] = [date, city, l3, forecast[-1]]

            # result_df.sort_values(by='Forecasted_Value', inplace=True, ascending=False,ignore_index=True)
            total_sum = result_df[f'{model_types[model_type]}_forecast'].sum()
            result_df[f'{model_types[model_type]}_popularity'] = np.round(
                result_df[f'{model_types[model_type]}_forecast'] / total_sum, 3)

            result_df.drop(f'{model_types[model_type]}_forecast', axis=1, inplace=True)
        except Exception as e:
            print(str(e))
        if len(final_df):
            final_df = pd.merge(final_df, result_df, on=['Date', 'City', 'L3_Category'])
        else:
            final_df = result_df

    wr = weights['revenue']
    wq = weights['quantity']

    final_df['Popularity'] = np.round((final_df['revenue_models_popularity'] * wr + final_df['quantity_models_popularity'] * wq) / (wr + wq), 3)

    final_df.drop([f'{model_types[_]}_popularity' for _ in model_types], axis=1, inplace=True)
    # print(final_df.columns)
    final_df.sort_values('Popularity', inplace=True, ascending=False)
    if not bottom:
        result_dict = final_df.iloc[:top_n_results][['L3_Category', 'Popularity']].set_index('L3_Category')[
            'Popularity'].to_dict()
    else:
        result_dict = final_df.iloc[-top_n_results:][['L3_Category', 'Popularity']].set_index('L3_Category')[
            'Popularity'].to_dict()
    # print(result_dict)
    return result_dict


# def generate_city_category_wise_top_products(city_col=None, model_file_path=None, city_value=None, l3_col=None,
#                                              l3_value=None, start_date=None, steps=None, max_model_date=None,
#                                              day_name_list=None, top_n_results=None,
#                                              category_product_contribution=None):
#     '''
#     city_col and max_model_date must be passed
#     pass max_model_date as string always
#     pass start_date as string
#     '''
#
#     final_dict = {}
#     err_dic = {}
#
#     if not start_date:
#         start_date = pd.to_datetime(dt.datetime.today().date())
#
#     else:
#         start_date = pd.to_datetime(start_date)
#
#     max_model_date = pd.to_datetime(max_model_date) + dt.timedelta(days=1)
#
#     gap_days = (start_date - max_model_date).days
#
#     steps += gap_days
#
#     end_date = max_model_date + dt.timedelta(days=steps - 1)
#
#     days = [_.day_name() for _ in pd.date_range(max_model_date, end_date)]
#
#     exog_array = [[check_for_days(day, day_name_list=day_name_list) for day in days]]
#
#     path = os.path.join(model_file_path, city_value)
#
#     model_dict = load_dict_from_pickle(file_name=f'{city_value}.pkl', dir_name=path)
#
#     key = f'{city_value}_{l3_value}'
#
#     model = model_dict.get(key, None)
#
#     try:
#         forecast = model.forecast(steps=steps, exog=exog_array, typ='levels')
#         date = str(forecast.index.date[-1])
#
#         for key in category_product_contribution:
#
#             if key[0] == city_value and key[1] == l3_value:
#                 final_dict[key[2]] = np.round(forecast[0] * category_product_contribution[key], 2)
#
#         final_dict = dict(
#             sorted([(a, b) for (a, b) in final_dict.items()][:top_n_results], key=lambda x: x[1], reverse=True))
#
#         final_dict = {i : final_dict[i] for i in final_dict if final_dict[i]!=0}
#
#         return final_dict
#     except:
#
#         err_dic['error_message'] = ('Make Sure Model Is Loaded Correctly')
#
#         return err_dic

## API2 Revised

def generate_city_category_wise_top_products(model_file_path=None, city_value=None, l3_col=None,
                                             l3_value=None, start_date=None, steps=None, max_model_date=None,
                                             day_name_list=None, top_n_results=None, contribution_file_path=None,
                                             model_types=None, weights=None, bottom=None):
    '''
    city_col and max_model_date must be passed
    pass max_model_date as string always
    pass start_date as string
    '''
    result_dic = {}
    #     final_dict={}
    err_dic = {}

    if not start_date:
        start_date = pd.to_datetime(dt.datetime.today().date())

    else:
        start_date = pd.to_datetime(start_date)

    max_model_date = pd.to_datetime(max_model_date) + dt.timedelta(days=1)

    gap_days = (start_date - max_model_date).days

    steps += gap_days

    end_date = max_model_date + dt.timedelta(days=steps - 1)

    days = [_.day_name() for _ in pd.date_range(max_model_date, end_date)]

    exog_array = [[check_for_days(day, day_name_list=day_name_list) for day in days]]

    for model_type in model_types:

        final_dict = {}

        new_model_file_path = os.path.join(model_file_path, model_types[model_type])

        new_contribution_file_path = os.path.join(contribution_file_path, f'{model_type}_wise')

        path = os.path.join(new_model_file_path, city_value)

        model_dict = load_dict_from_pickle(file_name=f'{city_value}.pkl', dir_name=path)

        contribution_dict = load_dict_from_pickle(file_name='l3_contributions', dir_name=new_contribution_file_path)

        key = f'{city_value}_{l3_value}'

        model = model_dict.get(key, None)

        try:
            forecast = model.forecast(steps=steps, exog=exog_array, typ='levels')

            date = str(forecast.index.date[-1])

            for key in contribution_dict:

                if key[0] == city_value and key[1] == l3_value:
                    final_dict[key[2]] = np.round(forecast[0] * contribution_dict[key], 2)

            total_sum = sum(final_dict.values())

            result_dic[model_type] = {a: np.round(b / total_sum, 3) for a, b in final_dict.items()}

        except:

            err_dic['error_message'] = ('Make Sure Model Is Loaded Correctly')

    popularity_dic = {}

    for models in result_dic:

        for products in result_dic[models]:
            popularity_dic[products] = popularity_dic.get(products, 0) + result_dic[models][products] * weights[models]

    if bottom:
        popularity_dic = dict(
            sorted([(a, np.round(b, 3)) for (a, b) in popularity_dic.items()], key=lambda x: x[1], reverse=True)[
            -top_n_results:])
    else:
        popularity_dic = dict(
            sorted([(a, np.round(b, 3)) for (a, b) in popularity_dic.items()], key=lambda x: x[1], reverse=True)[
            :top_n_results])

    return popularity_dic

################## TESTING ####################

# START_DATE ='2021-10-19'
# MAX_MODEL_DATE = '2021-10-18'
# DAY_NAME_LIST = ['Monday', 'Thursday']
# CONTRIBUTION_DIR = '/Users/utkarsh.lal/Desktop/forecasting_azure/forecasting/app/contributions'
# L3_COL = 'L3'
# MODEL_TYPES = {'revenue': 'revenue_models', 'quantity': 'quantity_models'}
# WEIGHTS = {'revenue': 0.7, 'quantity': 0.3}
# MODEL ='sarimax'
# MODEL_DIR = '/Users/utkarsh.lal/Desktop/forecasting_azure/forecasting/app/models'
# MODEL_FILE_PATH = f'{MODEL_DIR}/{MODEL}/trained_models/'
# l3_category = 'Noodles'
#
# df2 = generate_city_category_wise_top_products(model_file_path=MODEL_FILE_PATH, city_value='BANGALORE', start_date=START_DATE, steps=1, max_model_date=MAX_MODEL_DATE,\
#                                                  day_name_list=DAY_NAME_LIST, top_n_results=10, l3_col=L3_COL, l3_value=l3_category,\
#                                                  model_types=MODEL_TYPES, bottom=False, weights=WEIGHTS, contribution_file_path=CONTRIBUTION_DIR)






def generate_city_wise_top_categories_and_products(model_file_path=None, city_value=None,
                                                   start_date=None, steps=None, max_model_date=None,
                                                   day_name_list=None, top_n_results=None,
                                                   contribution_file_path=None, l3_col=None, model_types=None,
                                                   bottom=None, weights=None):
    final_dict = {}

    top_categories = generate_city_wise_top_categories(model_file_path=model_file_path, city_value=city_value,
                                                       start_date=start_date, steps=steps,
                                                       max_model_date=max_model_date, day_name_list=day_name_list,
                                                       top_n_results=top_n_results,
                                                       model_types=model_types, weights=weights, bottom=bottom)

    for category in top_categories:
        final_dict[category] = generate_city_category_wise_top_products(model_file_path=model_file_path,
                                                                        city_value=city_value, l3_col=l3_col,
                                                                        l3_value=category, start_date=start_date,
                                                                        steps=steps, max_model_date=max_model_date,
                                                                        day_name_list=day_name_list,
                                                                        top_n_results=top_n_results,
                                                                        contribution_file_path=contribution_file_path,
                                                                        model_types=model_types, weights=weights,
                                                                        bottom=bottom)

    return final_dict


def sarimax_prediction_configs(model_type=None):
    '''
    :return: compulsory configs for running APIs
    '''

    if model_type:
        basis_col = f"{model_type.split('_')[0]}_wise"
        dir_name = os.path.join(cfg.CONTRIBUTION_DIR, basis_col)
        category_product_contribution = load_dict_from_pickle(file_name=cfg.CONTRIBUTION_FILE_NAMES_MAP[cfg.L3_COL],
                                                              dir_name=dir_name)
        config_map = {
            'city_col': cfg.CITY_COL,
            'model_file_path': cfg.MODEL_FILE_PATH,
            'l3_col': cfg.L3_COL,
            'start_date': cfg.START_DATE,
            'max_model_date': cfg.MAX_MODEL_DATE,
            'day_name_list': cfg.DAY_NAME_LIST,
            'model_types': cfg.MODEL_TYPES,
            'category_product_contribution': category_product_contribution}

    else:
        config_map = {
            'city_col': cfg.CITY_COL,
            'model_file_path': cfg.MODEL_FILE_PATH,
            'l3_col': cfg.L3_COL,
            'start_date': cfg.START_DATE,
            'max_model_date': cfg.MAX_MODEL_DATE,
            'day_name_list': cfg.DAY_NAME_LIST,
            'model_types': cfg.MODEL_TYPES,
            'category_product_contribution': None

        }

    return config_map


## makes product contribution across categories in each city
## must be run after every model training

from pathlib import Path
# def get_project_root() -> Path:
#    return Path(__file__).parent.parent.parent

def save_city_l3_contribution_files():

    ROOT_DIR = Path(os.getcwd())
    ROOT_DIR = str(ROOT_DIR)
    ROOT_DIR = ROOT_DIR.replace('\\', "/")
    RAW_DATA_FILE_NAME = 'cleaned_data_2020-10-01_2021-10-18.parquet'
    data = pd.read_parquet(f"{ROOT_DIR}/datasets/cleaned_data_2020-10-01_2021-10-18.parquet")

    TARGET_COLS = {'revenue': 'Net_Sales', 'quantity': 'Net_Quantity'}
    L1_COL = 'L1'
    L2_COL = 'L2'
    L3_COL = 'L3'
    THRESHOLD_MIN_DATE = '2021-07-01'
    ARTICLE_DESCRIPTION_COL = 'SAP Article Description'
    CITY_COL = 'FC City'
    DATE_COL = 'Date'
    # CONTRIBUTION_DIR = 'contributions/'
    CONTRIBUTION_DIR = f"{ROOT_DIR}/contributions/"
    
    CONTRIBUTION_FILE_NAMES_MAP = {L3_COL: 'l3_contributions', L2_COL: 'l2_contributions', L1_COL: 'l1_contributions'}

    #data = pd.read_parquet(f"{ROOT_DIR}/datasets/cleaned_data_2020-10-01_2021-10-18.parquet")

    for key, value in TARGET_COLS.items():
        city_l3_article_contribution_map = get_level_wise_contribution(data, threshold_min_date=THRESHOLD_MIN_DATE,
                                                                       target_col=value,
                                                                       lower_level_cols=[ARTICLE_DESCRIPTION_COL],
                                                                       higher_level_cols=[CITY_COL, L3_COL],
                                                                       date_col=DATE_COL)

        dir_path = create_directory(root_path=CONTRIBUTION_DIR, directory_name=f'{key}_wise')
        save_dict_as_pickle(city_l3_article_contribution_map, file_name=CONTRIBUTION_FILE_NAMES_MAP[L3_COL],
                            dir_name=dir_path)


save_city_l3_contribution_files()


## Product level Popularity Functions

def generateProductPopularity(domain_id=None, product_skuid_list=None, city_name=None, popularity_dir=None,
                              w_q=0.7, w_r=0.3):
    quantity_popularity_dict = load_dict_from_pickle(file_name='quantity_wise_popularity.pkl', dir_name=popularity_dir)
    revenue_popularity_dict = load_dict_from_pickle(file_name='revenue_wise_popularity.pkl', dir_name=popularity_dir)

    final_dict = {}
    for product in product_skuid_list:
        final_dict[product] = np.round(
            (w_r * revenue_popularity_dict[product] + w_q * quantity_popularity_dict[product]) / (w_r + w_q), 3)

    return final_dict




##################### Testing ##########################

# START_DATE ='2021-10-19'
# MAX_MODEL_DATE = '2021-10-18'
# DAY_NAME_LIST = ['Monday', 'Thursday']
# CONTRIBUTION_DIR = '/Users/utkarsh.lal/Desktop/forecasting_azure/forecasting/app/contributions'
# L3_COL = 'L3'
# MODEL_TYPES = {'revenue': 'revenue_models', 'quantity': 'quantity_models'}
# WEIGHTS = {'revenue': 0.7, 'quantity': 0.3}
# MODEL ='sarimax'
# MODEL_DIR = '/Users/utkarsh.lal/Desktop/forecasting_azure/forecasting/app/models'
# MODEL_FILE_PATH = f'{MODEL_DIR}/{MODEL}/trained_models/'
#
# df = generate_city_wise_top_categories(model_file_path=MODEL_FILE_PATH, city_value='AHMEDABAD', start_date=START_DATE, steps=1, max_model_date=MAX_MODEL_DATE,\
#                                                  day_name_list=DAY_NAME_LIST, top_n_results=10,
#                                                  model_types=MODEL_TYPES, bottom=False, weights=WEIGHTS)