from modelling_utils import train_daily_model_and_save
from app.data import config as dcfg
from app.models import config as mcfg
import app.config as cfg
from app.data.data_utils import DATA_LOADER,make_map_object
from app.logger import APP_LOGGER
import os
import warnings
warnings.filterwarnings('ignore')


def load_data_and_train_model():

    '''
    loads data and save trained models in the respective directories
    :return: None
    '''
    try:
        data = DATA_LOADER.load_data(file_path=cfg.DATASET_DIR,file_name=cfg.RAW_DATA_FILE_NAME)
        APP_LOGGER.info(f'data loaded from {os.path.join(cfg.DATASET_DIR , cfg.RAW_DATA_FILE_NAME)}')

        map_data = DATA_LOADER.load_data(file_path=cfg.DATASET_DIR, file_name=cfg.MAP_FILE_NAME)

        l3_l2_map = make_map_object(map_data, key_col=dcfg.l3_col, val_col=dcfg.l2_col)

        l3_l1_map = make_map_object(map_data, key_col=dcfg.l3_col, val_col=dcfg.l1_col)

        APP_LOGGER.info('Starting Model Training')

        train_daily_model_and_save(data, model_dir_path=cfg.models_dir_path , l3_l2_map=l3_l2_map, l3_l1_map=l3_l1_map,
                               error_file_name='training_errors.csv', error_dir=cfg.TRAINING_ERROR_DIR, stats_and_visualization=False,
                               metric_file_name='metric.pkl'
                               , metric_file_dir=cfg.METRIC_DIR,city_col=dcfg.city_col, n_days=mcfg.n_days,test_size=mcfg.test_size
                                   ,y_col=dcfg.target_col, date_col=dcfg.date_col,
                               l3_col=dcfg.l3_col,l2_col=dcfg.l2_col, l1_col=dcfg.l1_col,
                               l3_groupby_col=mcfg.l3_groupby_col,l2_groupby_col=mcfg.l2_groupby_col, l1_groupby_col=mcfg.l1_groupby_col,seasonal=mcfg.seasonal)

    except:

        APP_LOGGER.exception('Error In Model Training file')


    return


if __name__ =='__main__':
    load_data_and_train_model()
