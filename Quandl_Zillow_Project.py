from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import dump, load
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log
from statsmodels.tsa.stattools import adfuller
import statsmodels
import datetime
import streamlit as st
from bs4 import BeautifulSoup
import requests
import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import nasdaqdatalink
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

warnings.filterwarnings('ignore')  # ignores warnings

# %% Connect via API
file_dir = 'C:/Users/timtu/PycharmProjects/web-scraping/'
# read in api key file from csv
df_api_keys = pd.read_csv(file_dir + 'api_keys.csv')
# get keys
quandl_api_key = df_api_keys.loc[df_api_keys['API'] == 'quandl']['KEY'].iloc[
    0]  # get the Key column' value at the location where API = quandl
# enter your key here
nasdaqdatalink.ApiConfig.api_key = quandl_api_key

# %% Load Quandl Zillow Data
# load zillow data (api gets hit, and it only pulls the first 10k when it does work, this link gives full data: https://data.nasdaq.com/tables/ZILLOW-DATA/export?api_key=XCGFdeqrL4E3XB3gdkij)
# zillow_data = nasdaqdatalink.get_table('ZILLOW/DATA', paginate=True)
zillow_data = pd.read_csv('ZILLOW_DATA_06262022.csv')
# due to size of the data, just keep .1% for now
zillow_data = zillow_data.sample(frac=.001)
# TODO change to larger dataset later instead of .1% subset
zillow_indicators = nasdaqdatalink.get_table(
    'ZILLOW/INDICATORS', paginate=True)
# store the region data in csv as backup, so we don't have to do API calls frequently on this.
# If we later want to reload, uncomment the first and second lines, change the csv date on the 2nd line, comment out the third line
# If we later want to redo this storage in csv, comment out the first and second lines, change the csv date on the 3rd line based on the updated filename, reload
# zillow_regions = nasdaqdatalink.get_table('ZILLOW/REGIONS', paginate=True)
# zillow_regions.to_csv('zillow_regions_06262022.csv', index=False)
zillow_regions = pd.read_csv('zillow_regions_06262022.csv')

# %% Explore Zillow data

# see average values, missing values
# zillow_data.groupby(['indicator_id', 'region_id'])['value'].median().head(2)  # get median value by indicator and region
zillow_indicators.isnull().values.any()  # check if any values are null
zillow_regions.isnull().sum().sum()  # check number of null values

# check unique values of indicators (56)
zillow_indicators['indicator'].nunique()
zillow_data['indicator_id'].nunique()

# plot some data
#plt.scatter(x='date', y='value', data=zillow_data)
#sns.catplot(x='region_id', y='value', data=zillow_data)


# %% Define functions for getting data from the region column:
# find state in region column
def check_state_in_str(search_str):
    search_str_list = [stg.strip() for stg in
                       search_str.split(';')]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if stg in states:  # if the string is in the list of states
            return stg  # store that state value


# find county in region column
def check_county_in_str(search_str):
    search_str_list = [stg.strip() for stg in
                       search_str.split(';')]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if 'county' in stg.lower():  # if the name "county" is in the lowercased string
            return stg  # store that county value


# find city in region column
list_of_cities = pd.read_csv('us_cities_states_counties.csv')[
    ['City', 'City alias']]  # load list of cities from github external source which was downloaded


def check_city_in_str(search_str):
    search_str_list = [stg.strip() for stg in
                       search_str.split(';')]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in the list
        if stg in list_of_cities.values:  # if that string is a value in the list of cities df
            return stg  # return that string


# Find metro area in region column
# Create list of metros from wiki page
# this wiki page has table of metro areas
metro_URL = 'https://en.wikipedia.org/wiki/Metropolitan_statistical_area'
res = requests.get(metro_URL)  # get the request object of this URL
try:  # check if the URL request was successful
    res.raise_for_status()
except Exception as exc:
    print('there was a problem: %s' % exc)
# create beautiful soup object of request html text string
soup = BeautifulSoup(res.text, 'lxml')
# elems = soup.select('#mw-content-text > div.mw-parser-output > table:nth-child(19)')  #select the html? maybe not necessary, continue below

metro_list = []  # initialize the list of metros
# we see when inspecting the wiki page, each row in the table is tagged by 'tr'
# The first row is a header, but loop through all remaining rows in this soup object
# rang(0,384)
for row in range(len(soup.find('table', class_='wikitable').find_all('tr')[1:])):
    # This selector is found by going to the inspect page, right clicking on the html that highlights the metropolitan area table
    html_data = soup.select(
        '#mw-content-text > div.mw-parser-output > table:nth-child(19) > tbody > tr:nth-child({})'.format(
            row + 2))  # get the data for each row in the table, where we replace the nth-child with the specific items iteration (row+2 to bypass the header).
    relevant_html_text = html_data[
        0].getText()  # get the key text of the html, which includes the metro area in the 1st name of the string
    try:
        metro_list.append(re.search('^\\n\d+\\n\\n(.*?),', relevant_html_text).group(
            1))  # find the metro text, which comes between the \n (1-many digit) \n\n and the comma
    except AttributeError:
        print('Could not find attribute in row {}, proceed'.format(
            row))  # in case the above can't find a proper value and throws attribute error, output the exception statement
    # re.search('^\\n\d\\n\\n(.*?),',soup.select('#mw-content-text > div.mw-parser-output > table:nth-child(19) > tbody > tr:nth-child({})'.format(6))[0].getText()).group(1)

# show the first and last few entries to make sure it worked
metro_list[0:4]
metro_list[len(metro_list) - 4:len(metro_list)]

# convert metro list to csv for storage
# create 1 key dictionary to list all the metros
metro_dict = {'metros': metro_list}
metro_list_df = pd.DataFrame(metro_dict)  # convert the dictionary to a df
metro_list_df.to_csv('metro_list.csv', index=False)  # output the df to a csv


# apply function to metro list to check if region's substring is a metro region
def check_metro_in_str(search_str):
    search_str_list = [stg.strip() for stg in
                       search_str.split(';')]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if stg in metro_list:  # if the string is in the list of metros
            return stg  # store that metro value


# %% Clean up the regions data, specifically its column
# check length of original regions dataset
len(zillow_regions)  # 79252
# check distribution of region types, i.e. the lowest level data is available at
zillow_regions.value_counts('region_type')

# keep only the regions which go down to the zip level, which is the most frequent
zillow_regions_new = zillow_regions.loc[zillow_regions['region_type'] == 'zip']
len(zillow_regions_new)  # 31189

# TODO possibly stick with zillow_regions, remove zillow_regions_new, since including non-zip region data should be fine since we don't rely on string length anymore

# split up the region column values into separate columns
# all states
states = ['IA', 'KS', 'UT', 'VA', 'NC', 'NE', 'SD', 'AL', 'ID', 'FM', 'DE', 'AK', 'CT', 'PR', 'NM', 'MS', 'PW', 'CO',
          'NJ', 'FL', 'MN', 'VI', 'NV', 'AZ', 'WI', 'ND', 'PA', 'OK', 'KY', 'RI', 'NH', 'MO', 'ME', 'VT', 'GA', 'GU',
          'AS', 'NY', 'CA', 'HI', 'IL', 'TN', 'MA', 'OH', 'MD', 'MI', 'WY', 'WA', 'OR', 'MH', 'SC', 'IN', 'LA', 'MP',
          'DC', 'MT', 'AR', 'WV', 'TX']

# extract the content from the region's column, note city column takes a while to generate, and returns false positives when metro name and city names overlap
zillow_regions_new.loc[:, 'region_str_len'] = zillow_regions_new.apply(lambda x: len(x['region'].split(';')),
                                                                       axis=1)  # split the region column by ;, get the length of that object, store it in a new column called region_str_len
zillow_regions_new.loc[:, 'zip_code'] = zillow_regions_new.apply(lambda x: re.search('(\d{5})', x['region']).group(),
                                                                 axis=1)  # find the 5 digit element in the region column, store it in a new column zip_code
zillow_regions_new.loc[:, 'state'] = zillow_regions_new.apply(lambda x: check_state_in_str(x['region']),
                                                              axis=1)  # find state in the region column, store it in state column
zillow_regions_new.loc[:, 'county'] = zillow_regions_new.apply(lambda x: check_county_in_str(x['region']),
                                                               axis=1)  # find county in the region column, store it in county column
zillow_regions_new.loc[:, 'city'] = zillow_regions_new.apply(lambda x: check_city_in_str(x['region']),
                                                             axis=1)  # find city in the region column, store it in city column
zillow_regions_new.loc[:, 'metro'] = zillow_regions_new.apply(lambda x: check_metro_in_str(x['region']),
                                                              axis=1)  # find metro in the regions column, store it in metro column

# %% Merge and clean all 3 tables for future analysis

# merge tables, using primary keys: https://data.nasdaq.com/databases/ZILLOW/documentation
# use inner join for both because I only care where the e.g. data's region and region's region are both available, and e.g. the data's indicator and indicator's indicator are both available
zillow_all = zillow_data.merge(zillow_regions_new, on='region_id').merge(
    zillow_indicators, on='indicator_id')

# remove certain columns to clean up and reduce size, sort by date:
zillow_all = zillow_all.drop(columns=['region_type', 'region_str_len'])
zillow_all = zillow_all.sort_values(by=['date'], ascending=False)
zillow_all.to_csv('zillow_all_backup.csv', index=False)

# %% More analysis in tableau
# TODO integrate tableau analysis into github, cleanup file below (which isn't that useful), display in github
zillow_all.value_counts('category')
zillow_all.value_counts('indicator')
zillow_all.info()

# %% Time series regression to predict next year's prices - preparing for model running
sns.set()

# set the font sizes of the plots:
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot the median price vs. date data
zillow_all.value = pd.to_numeric(
    zillow_all['value'])  # convert value to numeric
zillow_all.date = pd.to_datetime(
    zillow_all['date'], format='%Y-%m-%d')  # convert date to date
zillow_all.region_id = zillow_all['region_id'].astype(str)
zillow_price_by_date = zillow_all.groupby(['date'], as_index=False).median([
    'value'])  # this is our time series data
zillow_price_by_date.info()
zillow_price_by_date.describe().T

# plot data
#plt.rcParams['font.size'] = '24'
plt.ylabel('Median Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(zillow_price_by_date['date'], zillow_price_by_date['value'])
plt.show(block=True)

# check if data is stationary by Augmented Dickey Fuller test
adfuller_test = adfuller(zillow_price_by_date.value.dropna())
# p-value = .99, cannot reject Ho that time series is non-stationary
print('p-value: ', adfuller_test[1])

# plot differencing, autocorrelation - first differencing provides stationarity
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(zillow_price_by_date.value)
axes[0, 0].set_title('Original Series')
plot_acf(zillow_price_by_date.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(zillow_price_by_date.value.diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(zillow_price_by_date.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(zillow_price_by_date.value.diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(zillow_price_by_date.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# as per above, stationarity reached with 1 differencing, tentatively fix order of differencing as 1
# Adf test
ndiffs(zillow_price_by_date.value, test='adf')  # 1
# KPSS test
ndiffs(zillow_price_by_date.value, test='kpss')  # 1
# PP test
ndiffs(zillow_price_by_date.value, test='pp')  # 0
# (1,1,0)

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(zillow_price_by_date.value.diff())
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 5))
plot_pacf(zillow_price_by_date.value.diff().dropna(), ax=axes[1])
plt.show()

# ACF plot of 1st differenced series
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(zillow_price_by_date.value.diff())
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 5))
plot_acf(zillow_price_by_date.value.diff().dropna(), ax=axes[1])
plt.show()


# 1,1,0 ARIMA Model - seems from above analysis 1,1,0 was right
ARIMA_model = ARIMA(zillow_price_by_date.value, order=(1, 1, 0))
ARIMA_model_fit = ARIMA_model.fit()
print(ARIMA_model_fit.summary())

# Plot residual errors - no real pattern, constant mean and variance which is good
residuals = pd.DataFrame(ARIMA_model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Plot Actual vs Fitted
plot_predict(ARIMA_model_fit, dynamic=False)
plt.show()

# split price data into train and test
train = zillow_price_by_date.value[:260]
test = zillow_price_by_date.value[260:]

# Build Model - try different orders to get better forecasts in the plot. 1,2,1 has ok results...still linear though
ARIMA_model_train = ARIMA(train, order=(1, 2, 1))
ARIMA_fitted_train = ARIMA_model_train.fit()

# Forecast all linear for some reason...
# create forecast for len(test.index) items
fc = ARIMA_fitted_train.forecast(steps=len(test.index))

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# use auto-arima to determine lowest AIC model

auto_arima_model = pm.auto_arima(zillow_price_by_date.value, start_p=1, start_q=1,
                                 test='adf',       # use adftest to find optimal 'd'
                                 max_p=3, max_q=3,  # maximum p and q
                                 m=1,              # frequency of series
                                 d=None,           # let model determine 'd'
                                 seasonal=False,   # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

print(auto_arima_model.summary())

# plot residuals - overall looks ok
auto_arima_model.plot_diagnostics(figsize=(7, 5))
plt.show()

# forecast using optimal model
# Forecast
n_periods = 24
# make predictions for the next 24 periods
fc, confint = auto_arima_model.predict(
    n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(zillow_price_by_date.value), len(
    zillow_price_by_date.value)+n_periods)  # store index values of forecasted data

# make series for plotting purpose
# turn fc array into fc_series pandas series
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(zillow_price_by_date.value)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()

# Build SARIMA model to capture seasonal differences

# Seasonal - fit stepwise auto-ARIMA
zillow_price_by_date = zillow_price_by_date.set_index(
    'date')  # make the date an index for sarima to work

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(zillow_price_by_date[:], label='Original Series')
axes[0].plot(zillow_price_by_date[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasinal Differencing
axes[1].plot(zillow_price_by_date[:], label='Original Series')
axes[1].plot(zillow_price_by_date[:].diff(
    12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('House Prices', fontsize=16)
plt.show()

# fit the sarima model
auto_sarima_model = pm.auto_arima(zillow_price_by_date, start_p=1, start_q=1,
                                  test='adf',
                                  max_p=3, max_q=3, m=12,
                                  start_P=0, seasonal=True,
                                  d=None, D=1, trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

# show summary - best model was SARIMAX(1, 1, 2)x(0, 1, [1], 12)
auto_sarima_model.summary()

# Forecast next 24 months
n_periods = 24
fitted, confint = auto_sarima_model.predict(
    n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(
    zillow_price_by_date.index[-1], periods=n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(zillow_price_by_date)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of Housing Prices")
plt.show()


# use skforecast to forecast data...https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html
# Data manipulation
# ==============================================================================

# Plots
# ==============================================================================
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Modeling and Forecasting
# ==============================================================================


# Warnings configuration
# ==============================================================================
# warnings.filterwarnings('ignore')

# set date as index if not already, sort by date
zillow_price_by_date = zillow_price_by_date.set_index('date')
zillow_price_by_date = zillow_price_by_date.sort_index()

# Split data into train-test
# ==============================================================================
steps = 36
data_train = zillow_price_by_date[:-steps]  # train is all data up to n-36
data_test = zillow_price_by_date[-steps:]  # test is the final 36

print(
    f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(
    f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

fig, ax = plt.subplots(figsize=(9, 4))
data_train['value'].plot(ax=ax, label='train')
data_test['value'].plot(ax=ax, label='test')
ax.legend()
plt.show()

# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=6
)

forecaster.fit(y=data_train['value'])
forecaster

# Predictions...have to do some manipulation to include dates, give proper format
# ==============================================================================
steps = 36
predictions = pd.DataFrame([forecaster.predict(steps=steps)[:steps]]).T
len(predictions) == len(data_test.index)
predictions['date'] = data_test.index
len(predictions) == len(data_test)
predictions = predictions.set_index('date')
predictions.head(5)

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train['value'].plot(ax=ax, label='train')
data_test['value'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend()
plt.show()

# Hyperparameter Grid search
# ==============================================================================
steps = 36
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=12  # This value will be replaced in the grid search
)

# Lags used as predictors
lags_grid = [10, 20]

# Regressor's hyperparameters
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

results_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=data_train['value'],
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=steps,
    refit=True,
    metric='mean_squared_error',
    initial_train_size=int(len(data_train)*0.5),
    fixed_train_size=False,
    return_best=True,
    verbose=False
)

results_grid


# Predictions
# ==============================================================================
predictions = pd.DataFrame([forecaster.predict(steps=steps)[:steps]]).T
len(predictions) == len(data_test.index)
predictions['date'] = data_test.index
len(predictions) == len(data_test)
predictions = predictions.set_index('date')
predictions.head(5)


# Test error
# ==============================================================================
error_mse = mean_squared_error(
    y_true=data_test['value'],
    y_pred=predictions
)

print(f"Test error (mse): {error_mse}")


# split price data into train (pre-2020) and test (post-2020)
train = zillow_price_by_date[zillow_price_by_date.date <
                             pd.to_datetime('2020-01-01', format='%Y-%m-%d')]
test = zillow_price_by_date[zillow_price_by_date.date >=
                            pd.to_datetime('2020-01-01', format='%Y-%m-%d')]


# plot the train and test data...for some reason dates are numeric instead of dates...
# TODO figure out why x-axis is numbers instead of dates --> was using index instead of date, had to convert the date to an index
plt.plot(train, color="black")
plt.plot(test, color="red")
plt.ylabel('House Price', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.title("Train/Test split for Housing Data", fontsize=24)
plt.show()

# define y as the train data's housing prices
y = train['value']

# %% ARMA model
# To define an ARMA (no trend, no seasonality) model with the SARIMAX class, we pass in the order parameters of (1, 0 ,1):
ARMAmodel = SARIMAX(y, order=(1, 0, 1))
# fit ARMA model
ARMAmodel = ARMAmodel.fit()
# generate predictions on test data using model
# predict 29 out-of-sample forecasts
y_pred = ARMAmodel.get_forecast(len(test.index))
# generate a 95% confidence interval on the above predictions
y_pred_df = y_pred.conf_int(alpha=0.05)
# use ARMA model to predict the values at test index 288-316
y_pred_df["Predictions"] = ARMAmodel.predict(
    start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color='green', label='Predictions')
plt.legend()
plt.show()

# calculate RMSE
arma_rmse = np.sqrt(mean_squared_error(
    test["value"].values, y_pred_df["Predictions"]))
print("RMSE: ", arma_rmse)


# %% ARIMA model
# To define an ARIMA (lagging, differencing, white noise) model with the ARIMA class, we pass in the order parameters of (2, 2 ,2):
ARIMAmodel = ARIMA(y, order=(2, 2, 2))
# fit ARMA model
ARIMAmodel = ARIMAmodel.fit()
# generate predictions on test data using model
# predict 29 out-of-sample forecasts
y_pred = ARIMAmodel.get_forecast(len(test.index))
# generate a 95% confidence interval on the above predictions
y_pred_df = y_pred.conf_int(alpha=0.05)
# use ARMA model to predict the values at test index 288-316
y_pred_df["Predictions"] = ARIMAmodel.predict(
    start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color='Yellow', label='ARIMA Predictions')
plt.legend()
plt.show()

# calculate RMSE
arima_rmse = np.sqrt(mean_squared_error(
    test["value"].values, y_pred_df["Predictions"]))
print("RMSE: ", arima_rmse)

# %% SARIMA - Seasonal ARIMA

# To define a Seasonal ARIMA (historic values, shock events, seasonality) model with the SARIMAX class, we pass in the order parameters of (5, 4 ,2):
SARIMAXmodel = SARIMAX(y, order=(5, 4, 2), seasonal_order=(2, 2, 2, 12))
# fit SARIMAX model
SARIMAXmodel = SARIMAXmodel.fit()
# generate predictions on test data using model
# predict 29 out-of-sample forecasts
y_pred = SARIMAXmodel.get_forecast(len(test.index))
# generate a 95% confidence interval on the above predictions
y_pred_df = y_pred.conf_int(alpha=0.05)
# use ARMA model to predict the values at test index 288-316
y_pred_df["Predictions"] = SARIMAXmodel.predict(
    start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color='Blue', label='SARIMAX Predictions')
plt.legend()
plt.show()

# calculate RMSE
sarimax_rmse = np.sqrt(mean_squared_error(
    test["value"].values, y_pred_df["Predictions"]))
print("RMSE: ", sarimax_rmse)
