from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.models import config as mcfg
from app.data.data_management import *
from app.data.data_utils import *
from app.logger import APP_LOGGER

## Grid Search


def exp_smoothing_forecast(data, config):
    t, s, p, r = config
    # define model
    fitted_model = ExponentialSmoothing(data, trend=t, seasonal=s, seasonal_periods=p).fit(remove_bias=r)

    return fitted_model


# root mean squared error or rmse
def measure_rmse(true, predictions):

    predictions[np.isnan(predictions)] = 0

    try:
        rmse = np.sqrt(mean_squared_error(true, predictions))
    except:
        rmse = np.NaN
    return rmse


def measure_r_squared(true, predictions):
    predictions[np.isnan(predictions)] = 0

    try:
        r_squared_score = r2_score(true, predictions)
    except:
        r_squared_score = np.NaN

    return r_squared_score


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    try:
        model = exp_smoothing_forecast(data, cfg)
    except:
        return np.NaN, np.NaN

    predictions = model.forecast(n_test)
    error = measure_rmse(test, predictions)
    r_squared_value = measure_r_squared(test, predictions)

    return error, r_squared_value


# score a model, return None on failure

def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # define config list
    t_params = ['add', 'mul', None]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for s in s_params:
            for p in p_params:
                for r in r_params:
                    cfg = [t, s, p, r]
                    models.append(cfg)
    return models


def run_daily_exp_smoothening_grid_search(df, n_days=None, y_col=None, date_col=None, test_size=None,seasonal=None):

    error = []
    r2_score = []
    mod = []
    df = df.iloc[-n_days:]
    data = df[y_col]
    n_test = test_size
    # model configs
    cfg_list = exp_smoothing_configs(seasonal=seasonal)

    for c in cfg_list:
        mod.append(c)
        res, r_squared = walk_forward_validation(data, n_test, c)
        error.append(res)
        r2_score.append(r_squared)

    index = error.index(min(error))
    model_dict = {'model': mod[index], 'r2_score': r2_score[index]}

    return model_dict


def fit_daily_expsmoothening_model(df, n_days=None, y_col=None, date_col=None, test_size=None, seasonal=None):
    df.set_index(date_col, inplace=True)
    #df.index.freq='D'

    ## converting zeroes to be 0.1 before passing to exponential smoothening model.
    df[y_col] = np.where(df[y_col] <= 0.0, 0.1, df[y_col])

    data = df[y_col]

    model_dict = run_daily_exp_smoothening_grid_search(df, n_days=n_days, y_col=y_col, date_col=date_col,
                                                       test_size=test_size,seasonal=seasonal)

    try:
        fitted_model = exp_smoothing_forecast(data, model_dict['model'])
    except:
        fitted_model = None

    return {'model': fitted_model, 'r2_score': model_dict['r2_score']}

def train_daily_model_and_save(df, model_dir_path=None, l3_l2_map=None, l3_l1_map=None,
                               error_file_name='errors.csv', error_dir=None, stats_and_visualization=False,
                               metric_file_name='metric.pkl'
                               , metric_file_dir=None, city_col=None, n_days=None, test_size=None, y_col=None, date_col=None,
                               l3_col=None, l2_col=None, l1_col=None,
                               l3_groupby_col=None, l2_groupby_col=None, l1_groupby_col=None, seasonal=None):
    '''
    train model for particular granularity and saves it
    '''
    df_city_l3_grouped = df.groupby(l3_groupby_col)[y_col].sum().reset_index()
    df_city_l2_grouped = df.groupby(l2_groupby_col)[y_col].sum().reset_index()
    df_city_l1_grouped = df.groupby(l1_groupby_col)[y_col].sum().reset_index()

    unique_cities = df_city_l3_grouped[city_col].unique()

    saved_l3_model_list = os.listdir(model_dir_path[l3_col])
    saved_l2_model_list = os.listdir(model_dir_path[l2_col])
    saved_l1_model_list = os.listdir(model_dir_path[l1_col])

    ## Error_Cols
    City_Error = []
    L3_Error = []

    r_squared_dict = load_dict_from_pickle(file_name=metric_file_name, dir_name=metric_file_dir)

    if not r_squared_dict:
        r_squared_dict = {}

    for city in unique_cities:
        df_city_l3 = df_city_l3_grouped[df_city_l3_grouped[city_col] == city].copy()
        df_city_l2 = df_city_l2_grouped[df_city_l2_grouped[city_col] == city].copy()
        df_city_l1 = df_city_l1_grouped[df_city_l1_grouped[city_col] == city].copy()

        unique_l3 = df_city_l3[l3_col].unique()

        for l3 in unique_l3:

            df_l3 = df_city_l3[df_city_l3[l3_col] == l3].iloc[-n_days:]
            df_l2 = df_city_l2[df_city_l2[l2_col] == l3_l2_map[l3]].iloc[-n_days:]
            df_l1 = df_city_l1[df_city_l1[l1_col] == l3_l1_map[l3]].iloc[-n_days:]

            if f'{city}_{l3}.pkl' in saved_l3_model_list or f'{city}_{l3_l2_map[l3]}.pkl' in saved_l2_model_list or f'{city}_{l3_l1_map[l3]}.pkl' in saved_l1_model_list:
                continue
            else:

                try:
                    if get_zero_count(df_l3, col= y_col) <= n_days//10:

                        model_dict = fit_daily_expsmoothening_model(df_l3, n_days=n_days, y_col=y_col,
                                                                    date_col=date_col, test_size=test_size,seasonal= seasonal)

                        save_trained_model(model_dict['model'], file_name=f'{city}_{l3}.pkl',
                                           dir_path=model_dir_path[l3_col])

                        r_squared_dict[f'{city}_{l3}'] = model_dict['r2_score']

                        if stats_and_visualization:
                            print(f'R-Squared value for model of {city}_{l3} is {model_dict["r2_score"]}')



                    elif get_zero_count(df_l2, col=y_col) <= n_days//10:

                        model_dict = fit_daily_expsmoothening_model(df_l2, n_days=n_days, y_col=y_col,
                                                                    date_col=date_col, test_size=test_size,seasonal=seasonal)

                        save_trained_model(model_dict['model'], file_name=f'{city}_{l3_l2_map[l3]}.pkl',
                                           dir_path=model_dir_path[l2_col])

                        r_squared_dict[f'{city}_{l3_l2_map[l3]}'] = model_dict['r2_score']
                        if stats_and_visualization:
                            print(f'R-Squared value for model of {city}_{l3_l2_map[l3]} is {model_dict["r2_score"]}')

                    elif get_zero_count(df_l1, col='Net_Sales') <= n_days//10:

                        model_dict = fit_daily_expsmoothening_model(df_l1, n_days=n_days, y_col=y_col,
                                                                   date_col=date_col, test_size=test_size,seasonal=seasonal)

                        save_trained_model(model_dict['model'], file_name=f'{city}_{l3_l1_map[l3]}.pkl',
                                           dir_path=model_dir_path[l1_col])

                        r_squared_dict[f'{city}_{l3_l1_map[l3]}'] = model_dict['r2_score']
                        if stats_and_visualization:
                            print(f'R-Squared value for model of {city}_{l3_l1_map[l3]} is {model_dict["r2_score"]}')





                    else:
                        r_squared_dict[f'{city}_{l3}'] = np.NaN

                        City_Error.append(city)
                        L3_Error.append(l3)
                except Exception as e:
                    APP_LOGGER.exception('Error in Model training Function : ' + str(e))
                    save_dict_as_pickle(r_squared_dict, file_name=metric_file_name, dir_name=metric_file_dir)
                    return
    try:
        df_error = pd.DataFrame({city_col: City_Error, l3_col: L3_Error})
        df_error['Error_Description'] = mcfg.error_message

        save_dict_as_pickle(r_squared_dict, file_name=metric_file_name, dir_name=metric_file_dir)
        save_df_as_csv(df_error, file_name=error_file_name, dir_name=error_dir)
    except Exception as e:
        APP_LOGGER.exception('Error in Saving models and metrices  : ' + str(e))
        save_dict_as_pickle(r_squared_dict, file_name=metric_file_name, dir_name=metric_file_dir)
        return

    return

