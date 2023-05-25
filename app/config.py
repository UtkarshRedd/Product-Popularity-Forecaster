import os

import pathlib
#import app
# from app.data.config import l1_col,l2_col,l3_col

IS_CUSTOM_ENV = True
IS_PROD_ENV = False

from pathlib import Path
def get_project_root() -> Path:
   return Path(__file__).parent.parent

#print(get_project_root())

THRESHOLD_MIN_DATE = '2021-07-01'

MAX_MODEL_DATE = '2021-10-18'

START_DATE = '2021-10-19'
'''
COMMON COLS
'''

L3_COL = 'L3'
L2_COL = 'L2'
L1_COL = 'L1'

CITY_COL='FC City'
DATE_COL ='Date'

TARGET_COLS = {'revenue': 'Net_Sales', 'quantity': 'Net_Quantity'}

ARTICLE_ID_COL = 'SAP Article Id'
ARTICLE_DESCRIPTION_COL = 'SAP Article Description'

# TARGET_COL = 'Net_Sales'

N_DAYS=60

SEASONAL_PERIOD = 7
EXOG_COL = 'Monday_or_Thursday'



L1_GROUPBY_COLS=[CITY_COL,L1_COL,DATE_COL]
L2_GROUPBY_COLS=[CITY_COL,L2_COL,DATE_COL]
L3_GROUPBY_COLS=[CITY_COL,L3_COL,DATE_COL]


DAY_NAME_COL= 'Day_Name'
DAY_NAME_LIST = ['Monday', 'Thursday']

# PACKAGE_ROOT = pathlib.Path(app.__file__).resolve().parent
# PROJECT_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = get_project_root()
PACKAGE_ROOT = PROJECT_ROOT
#print(PROJECT_ROOT)
# directory names
DATASET_DIR = f'{PROJECT_ROOT}/datasets'
LOG_DIR = f'{PROJECT_ROOT}/logs'
METRIC_DIR = f'{PACKAGE_ROOT}/metrics'
ERROR_DIR = f'{PACKAGE_ROOT}/errors'
FORECAST_DIR = f'{PACKAGE_ROOT}/saved_forecasts'
TRAINING_ERROR_DIR = f'{ERROR_DIR}/training_errors'
FORECAST_ERROR_DIR = f'{ERROR_DIR}/forecast_errors'
CONTRIBUTION_DIR = f'{PACKAGE_ROOT}/contributions'
POPULARITY_DIR = f'{PACKAGE_ROOT}/saved_popularity'


## file_names

RAW_DATA_FILE_NAME = 'cleaned_data_2020-10-01_2021-10-18.parquet'
MAP_FILE_NAME = 'L1 L2 L3 Mapping.xlsx'

# ### Model Directory Path
# models_dir = PACKAGE_ROOT/'trained_models'
# l3_models_path = models_dir/'l3_models'
# l2_models_path = models_dir/'l2_models'
# l1_models_path = models_dir/'l1_models'
#
# models_dir_path = {l1_col:l1_models_path,l2_col:l2_models_path,l3_col:l3_models_path}

MODEL_DIR = f'{PACKAGE_ROOT}/models'
'''
MODEL WISE CONFIGS
'''

MODEL = 'SARIMAX'

MODEL_TYPES ={'revenue': 'revenue_models', 'quantity': 'quantity_models'}
WEIGHTS = {'revenue': 0.7, 'quantity':0.3}


MODEL_FILE_PATH = f'{MODEL_DIR}/{MODEL}/trained_models'
ERROR_FILE_PATH = f'{MODEL_DIR}/{MODEL}/errors'
CONTRIBUTION_FILE_NAMES_MAP = {L3_COL: 'l3_contributions', L2_COL: 'l2_contributions', L1_COL: 'l1_contributions'}




