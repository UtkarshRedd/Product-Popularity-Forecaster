import sys
sys.path.append('/Users/utkarsh.lal/Desktop/forecasting_azure/forecasting/app/')
print(sys.path)
from forecasting.app.models.sarimax.sarimax_utils import (create_day_name_col,
                                              check_for_particular_day,
                                              train_city_l3_wise_sarimax_and_save)

import warnings

warnings.filterwarnings('ignore')


def train(df, day_name_col=None, day_name_list=None, city_col=None, target_col=None, l3_col=None, n_days=None,
          seasonal_period=None, date_col=None, exog_col=None,model_file_path=None, error_file_path=None):

    df_day = create_day_name_col(df, date_col)
    df_day = check_for_particular_day(df_day, day_name_col=day_name_col, day_name_list=day_name_list)

    error_df = train_city_l3_wise_sarimax_and_save(df_day, city_col=city_col, target_col=target_col, l3_col=l3_col,
                                                   n_days=n_days,
                                                   seasonal_period=seasonal_period, date_col=date_col,
                                                   exog_col=exog_col,model_file_path=model_file_path,error_file_path=error_file_path)

    return error_df



########## Testing #########
