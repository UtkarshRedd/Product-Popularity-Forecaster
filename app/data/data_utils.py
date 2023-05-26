import pandas as pd
import numpy as np
import os
import app.data.config as cfg


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_data_from_excel(file_path, file_name, skip_pages=0):
        '''
        ## Reads Excel File with single/multiple sheets and returns consolidated dataframe.
        ## skip_pages : The number of pages to skip from front.
        '''

        if file_path:
            file_name = os.path.join(file_path, file_name)

        xls = pd.ExcelFile(file_name)
        sheet_names = xls.sheet_names

        if len(sheet_names) > 1:
            df_final = pd.DataFrame()
            dict = pd.read_excel(file_name, sheet_name=None)
            for sheets in sheet_names[skip_pages:]:
                df = dict[sheets]
                df_final = df_final.append(df, ignore_index=True)

        else:

            df_final = pd.read_excel(file_name)

        return df_final

    @staticmethod
    def load_data_from_csv(file_path=None, file_name=None):
        '''
        ## Reads csv file and returns consolidated dataframe
        '''
        if file_path:
            file_name = os.path.join(file_path, file_name)

        df = pd.read_csv(file_name, encoding_errors='ignore')

        return df

    @staticmethod
    def load_data_from_parquet(file_path=None, file_name=None):
        '''
        ## Reads paraquet file and returns consolidated dataframe
        '''
        if file_path:
            file_name = os.path.join(file_path, file_name)

        df = pd.read_parquet(file_name)

        return df

    @staticmethod
    def load_data(file_path=None, file_name=None):
        '''
        :param file_path: directory path of the file
        :param file_name: name of the file
        :return: dataframe after reading the file
        '''

        if file_name.endswith('.csv'):
            df = DataLoader.load_data_from_csv(file_path=file_path, file_name=file_name)
        elif file_name.endswith('.xlsx'):
            df = DataLoader.load_data_from_excel(file_path=file_path, file_name=file_name)
        elif file_name.endswith('.parquet'):
            df = DataLoader.load_data_from_parquet(file_path=file_path, file_name=file_name)
        else:
            raise Exception('Invalid File Format')

        return df


def get_zero_count(df, col=None):
    '''
    returns the number of zeros in a particular column of a df
    '''

    return df[df[col] == 0.0].shape[0]

def make_map_object(df, key_col=None, val_col=None):
    '''
    makes a dictionary with rows of key col as keys and value col as values
    '''
    map_obj = df[[key_col, val_col]].set_index(key_col).to_dict()[val_col]

    return map_obj


'''
Function To Get Group By Aggregate Columns
'''


def aggregate_data_level_wise(df, groupby_cols=None, target_col=None):
    if groupby_cols and target_col:
        aggregated_df = df.groupby(groupby_cols)[target_col].sum().reset_index()

    else:
        raise Exception('pass required cols')

    return aggregated_df


def get_level_wise_contribution(df, threshold_min_date=None, target_col=None, higher_level_cols=None,
                                lower_level_cols=None,date_col=None):
    '''
    returns the contribution of lower granularity in higher granularity

    higher_level_cols:->list
    lower_level_cols:->list
    :param date_col:
    '''
    df = df[df[date_col] >= pd.to_datetime(threshold_min_date)]

    lower_grp = df.groupby(higher_level_cols + lower_level_cols)[target_col].sum()
    higher_grp = df.groupby(higher_level_cols)[target_col].sum()

    contribution_df = lower_grp / higher_grp

    contribution_df = contribution_df.reset_index()

    map_obj = contribution_df.set_index(higher_level_cols + lower_level_cols).to_dict()[target_col]

    return map_obj


def save_df_as_csv(df, file_name=None, dir_name=None):

    if dir_name:
        file_name = os.path.join(dir_name, file_name)

    df.to_csv(file_name, index=False)

    return

DATA_LOADER = DataLoader()


