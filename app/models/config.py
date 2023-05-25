
## modelling config for daily forecasts

n_days = 30 # last number of  days data used for daily forecasts
test_size = 7  # 7 days data used for testing purpose


## seasonal parameters
seasonal = [7,15,30]


## Groupby_cols
l3_groupby_col = ['FC City', 'L3', 'Date']
l2_groupby_col = ['FC City', 'L2', 'Date']
l1_groupby_col = ['FC City', 'L1', 'Date']


## error
error_message = 'Time Series Formed is very insignificant'


