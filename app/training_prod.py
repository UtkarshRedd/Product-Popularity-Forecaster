import os
import sys

import config as cfg
from models.sarimax.training import train as train_sarimax
from logger import APP_LOGGER
import warnings


warnings.filterwarnings('ignore')


from data.data_utils import aggregate_data_level_wise, DATA_LOADER


def train(model=cfg.MODEL):  #MODEL = 'SARIMAX' as declared in config file
    '''
        loads data and save trained models in the respective directories
        :return: None
        '''
    try:
        data = DATA_LOADER.load_data(file_path=cfg.DATASET_DIR, file_name=cfg.RAW_DATA_FILE_NAME)
        APP_LOGGER.info(f'data loaded from {os.path.join(cfg.DATASET_DIR, cfg.RAW_DATA_FILE_NAME)}')
        #
        # map_data = DATA_LOADER.load_data(file_path=cfg.DATASET_DIR, file_name=cfg.MAP_FILE_NAME)
        #
        # l3_l2_map = make_map_object(map_data, key_col=dcfg.l3_col, val_col=dcfg.l2_col)
        #
        # l3_l1_map = make_map_object(map_data, key_col=dcfg.l3_col, val_col=dcfg.l1_col)

        df_l3_wise = aggregate_data_level_wise(data, groupby_cols=cfg.L3_GROUPBY_COLS, target_col=cfg.TARGET_COL)

        APP_LOGGER.info('Starting Model Training')

        if model=='SARIMAX':
            train_sarimax(df_l3_wise, day_name_col=cfg.DAY_NAME_COL, day_name_list=cfg.DAY_NAME_LIST, city_col=cfg.CITY_COL, target_col=cfg.TARGET_COL, l3_col=cfg.L3_COL, n_days=cfg.N_DAYS,
              seasonal_period=cfg.SEASONAL_PERIOD, date_col=cfg.DATE_COL, exog_col=cfg.EXOG_COL, model_file_path=cfg.MODEL_FILE_PATH, error_file_path=cfg.ERROR_FILE_PATH)
        else:
            pass
    except:

        APP_LOGGER.exception('Error In Model Training file')


    return


if __name__ =='__main__':
    train()

###### Testing #####
df = train('SARIMAX')
