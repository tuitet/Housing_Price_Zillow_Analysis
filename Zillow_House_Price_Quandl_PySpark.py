# import streamlit as st
import datetime
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import nasdaqdatalink  # pip install nasdaq-data-link
import numpy as np
import pandas as pd
import pmdarima as pm
import pyspark
import requests
from bs4 import BeautifulSoup
from joblib import dump
from joblib import load
from numpy import log
from numpy.random import MT19937
from numpy.random import RandomState
from numpy.random import SeedSequence
from pmdarima.arima.utils import ndiffs
from prophet import Prophet
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.pandas.plot import matplotlib
from pyspark.sql import dataframe
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import coalesce
from pyspark.sql.functions import col
from pyspark.sql.functions import count
from pyspark.sql.functions import isnan
from pyspark.sql.functions import isnull
from pyspark.sql.functions import when
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import \
    ForecasterAutoregMultiOutput
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import \
    TimeSeriesSplit  # Splitting for time series CV!
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import seaborn as sns

# import statsmodels

warnings.filterwarnings("ignore")  # ignores warnings

# TODO convert to jupyter notebook? https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html#ui

# %% Imports and setup spark session

# avoid later errors? https://stackoverflow.com/questions/68705417/pycharm-error-java-io-ioexception-cannot-run-program-python3-createprocess

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
spark = SparkSession.builder.getOrCreate()

pyspark.__version__  # 3.3.0

# Create SparkSession object using all available cores on this local computer
spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

# What version of Spark? 3.3.0
print(spark.version)

# %% Connect via API
# set file directory where the api keys file is stored
file_dir = "C:/Users/timtu/PycharmProjects/PySpark/"
# read in api key file from csv
df_api_keys = pd.read_csv(file_dir + "api_keys.csv")
# get quandl key
quandl_api_key = df_api_keys.loc[df_api_keys["API"] == "quandl"]["KEY"].iloc[
    0]  # get the Key column' value at the location where API = quandl
# enter key here
nasdaqdatalink.ApiConfig.api_key = quandl_api_key

# %% Load Quandl Zillow Data
# load zillow data (api gets hit, and it only pulls the first 10k when it does work, this link gives full data: https://data.nasdaq.com/tables/ZILLOW-DATA/export?api_key=XCGFdeqrL4E3XB3gdkij)
# zillow_data = nasdaqdatalink.get_table('ZILLOW/DATA', paginate=True)
# Read data from CSV file
zillow_data = spark.read.csv("ZILLOW_DATA_12312022.csv",
                             sep=",",
                             header=True,
                             inferSchema=True,
                             nullValue="NA")

# check basics of the dataset
zillow_data.show(2)
zillow_data.printSchema()
print(zillow_data.dtypes)
print("The data contain %d records." % zillow_data.count())  # 135,439,925 rows

# skip now that we're ready...
# zillow_data = zillow_data.sample(fraction=.001)  # due to size of the data, just keep .1% for now
# zillow_data.count() #133387
# TODO change to larger dataset later instead of .1% subset

zillow_indicators = nasdaqdatalink.get_table("ZILLOW/INDICATORS",
                                             paginate=True)
zillow_indicators_pyspark = spark.createDataFrame(zillow_indicators)
try:
    zillow_indicators_pyspark.show(1)
except:
    print("proceed, previously got pycharm java python error")

# store the region data in csv as backup, so we don't have to do API calls frequently on this.
# If we later want to reload, uncomment the first and second lines below, change the csv date on the 2nd line to the current date, comment out the third line
# If we later want to redo this storage in csv, comment out the first and second lines, change the csv date on the 3rd line based on the updated filename, reload
zillow_regions = nasdaqdatalink.get_table("ZILLOW/REGIONS", paginate=True)
zillow_regions.to_csv("zillow_regions_12312022.csv", index=False)
# load regions data from stored csv, mixture of comma and semicolon as separator so let it infer
zillow_regions = spark.read.csv("zillow_regions_12312022.csv",
                                header=True,
                                inferSchema=True,
                                nullValue="NA")
# zillow_regions.show(2)

# %% Explore Zillow data

# play around with pyspark data check options
zillow_indicators.isnull().values.any()  # check if any values are null 0
# check number of null values 0
zillow_regions.filter("region_id IS NULL").count()
# create a Table/View from the PySpark DataFrame to use sql queries on
zillow_data.createOrReplaceTempView("zillow_data_temp")
# spark.sql('select count(distinct(*)) from zillow_data_temp').show(2) #count distinct rows in zillow_data
# zillow_indicators.indicator_id.unique()
# zillow_indicators.category.value_counts()
# spark.sql('select indicator_id, count(*) from zillow_data_temp group by indicator_id').show()
# show distinct indicator values
zillow_data.select("indicator_id").distinct().show()
# select mean value of all  single family homes (indicator_id like Z) by region
zillow_indicators.columns
zillow_indicators[["indicator_id", "indicator"]]
spark.sql("SELECT indicator_id, MEAN(value) "
          "FROM zillow_data_temp "
          "WHERE indicator_id LIKE 'R%' "
          "GROUP BY indicator_id LIMIT 5").show()


# %% Define functions for getting data from the region column:
# find state in region column
def check_state_in_str(search_str):
    search_str_list = [stg.strip() for stg in search_str.split(";")
                       ]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if stg in states:  # if the string is in the list of states
            return stg  # store that state value


# find county in region column
def check_county_in_str(search_str):
    search_str_list = [stg.strip() for stg in search_str.split(";")
                       ]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if "county" in stg.lower(
        ):  # if the name "county" is in the lowercased string
            return stg  # store that county value


# find city in region column
list_of_cities = pd.read_csv("us_cities_states_counties.csv")[[
    "City", "City alias"
]]  # load list of cities from github external source which was downloaded


def check_city_in_str(search_str):
    search_str_list = [stg.strip() for stg in search_str.split(";")
                       ]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in the list
        if (stg in list_of_cities.values
            ):  # if that string is a value in the list of cities df
            return stg  # return that string


# Find metro area in region column
# Create list of metros from wiki page
# this wiki page has table of metro areas
metro_URL = "https://en.wikipedia.org/wiki/Metropolitan_statistical_area"
res = requests.get(metro_URL)  # get the request object of this URL
try:  # check if the URL request was successful
    res.raise_for_status()
except Exception as exc:
    print("there was a problem: %s" % exc)
# create beautiful soup object of request html text string
soup = BeautifulSoup(res.text)
# elems = soup.select('#mw-content-text > div.mw-parser-output > table:nth-child(19)')  #select the html? maybe not necessary, continue below

metro_list = []  # initialize the list of metros
# we see when inspecting the wiki page, each row in the table is tagged by 'tr'
# The first row is a header, but loop through all remaining rows in this soup object
# rang(0,384)
for row in range(len(
        soup.find("table", class_="wikitable").find_all("tr")[1:])):
    # This selector is found by going to the inspect page, right clicking on the html that highlights the metropolitan area table
    try:
        html_data = soup.select(
            "#mw-content-text > div.mw-parser-output > table:nth-child(19) > tbody > tr:nth-child({})"
            .format(row + 2)
        )  # get the data for each row in the table, where we replace the nth-child with the specific items iteration (row+2 to bypass the header).
        relevant_html_text = html_data[0].getText(
        )  # get the key text of the html, which includes the metro area in the 1st name of the string...runs into list index out of range error, can't proceed
        try:
            metro_list.append(
                re.search("^\\n\d+\\n\\n(.*?),", relevant_html_text).group(1)
            )  # find the metro text, which comes between the \n (1-many digit) \n\n and the comma
        except AttributeError:
            print(
                "Could not find attribute in row {}, proceed".format(row)
            )  # in case the above can't find a proper value and throws attribute error, output the exception statement
        # re.search('^\\n\d\\n\\n(.*?),',soup.select('#mw-content-text > div.mw-parser-output > table:nth-child(19) > tbody > tr:nth-child({})'.format(6))[0].getText()).group(1)
    except:
        print("just use metro backup csv below")
        break

# show the first and last few entries to make sure it worked
metro_list[0:4]
metro_list[len(metro_list) - 4:len(metro_list)]

# convert metro list to csv for storage
# create 1 key dictionary to list all the metros
metro_dict = {"metros": metro_list}
metro_list_df = pd.DataFrame(metro_dict)  # convert the dictionary to a df
metro_list_df.to_csv("metro_list.csv", index=False)  # output the df to a csv

metro_list_df_backup = pd.read_csv(
    "metro_list_df_backup1.csv")  # read back in the metro names

# apply function to metro list to check if region's substring is a metro region


def check_metro_in_str(search_str):
    search_str_list = [stg.strip() for stg in search_str.split(";")
                       ]  # split region column by semicolon, strip whitespace
    for stg in search_str_list:  # for each string in that list
        if stg in metro_list_df_backup:  # if the string is in the list of metros
            return stg  # store that metro value


# %% Clean up the regions data, specifically its column
# check length of original regions dataset
zillow_regions.count()  # 86098
# check the distinct values of region types, i.e. the levels data is available at
zillow_regions_distinct = zillow_regions.select("region_type")
zillow_regions_distinct = zillow_regions_distinct.distinct()
zillow_regions_distinct.show()

# keep only the regions which go down to the zip level, which is the most frequent
zillow_regions.createOrReplaceTempView("zillow_regions_temp")
zillow_regions_new = spark.sql(
    "SELECT * FROM zillow_regions_temp WHERE region_type = 'zip'")
# zillow_regions_new.show(1)
zillow_regions_new.count()  # 31204

# TODO possibly stick with zillow_regions, remove zillow_regions_new, since including non-zip region data should be fine since we don't rely on string length anymore

# split up the region column values into separate columns
# all states
states = [
    "IA",
    "KS",
    "UT",
    "VA",
    "NC",
    "NE",
    "SD",
    "AL",
    "ID",
    "FM",
    "DE",
    "AK",
    "CT",
    "PR",
    "NM",
    "MS",
    "PW",
    "CO",
    "NJ",
    "FL",
    "MN",
    "VI",
    "NV",
    "AZ",
    "WI",
    "ND",
    "PA",
    "OK",
    "KY",
    "RI",
    "NH",
    "MO",
    "ME",
    "VT",
    "GA",
    "GU",
    "AS",
    "NY",
    "CA",
    "HI",
    "IL",
    "TN",
    "MA",
    "OH",
    "MD",
    "MI",
    "WY",
    "WA",
    "OR",
    "MH",
    "SC",
    "IN",
    "LA",
    "MP",
    "DC",
    "MT",
    "AR",
    "WV",
    "TX",
]

# TODO Have to repurpose this from python to pyspark using withColumn and other approaches? (UDF)...or keep as pandas and later convert output back to pyspark?
# Extract the content from the region's column, note city column takes a while to generate, and returns false positives when metro name and city names overlap

# from pyspark.sql.functions import when, lit, col, udf
# udf_func_length = udf(lambda x: len(x['region'].split(';')))
# zillow_regions_new = zillow_regions_new.withColumn('region_str_len', udf_func_length(zillow_regions_new.region))
# zillow_regions_new.show()
# zillow_regions_new.loc[:, 'region_str_len'] = zillow_regions_new.apply(lambda x: len(x['region'].split(';')),
#                                                                        axis=1)  # split the region column by ;, get the length of that object, store it in a new column called region_str_len

# perform pandas operations on pyspark df
# import pyspark.pandas as ps #have to install pyarrow package
# ps.set_option('compute.ops_on_diff_frames', True)
# zillow_regions_new.show(1)
# for the below 2 lines - https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html
# optimizes conversion between spark df and pandas df via arrow
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# Ensures arrow is used even if errors occur
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
zillow_regions_new_pandas = zillow_regions_new.toPandas()
# zillow_regions_new = zillow_regions_new.withColumn('zip_code', lit(lambda x: re.search('(\d{5})', x['region'].group())))

zillow_regions_new_pandas.loc[:, "zip_code"] = zillow_regions_new_pandas.apply(
    lambda x: re.search("(\d{5})", x["region"]).group(), axis=1
)  # find the 5 digit element in the region column, store it in a new column zip_code

zillow_regions_new_pandas.loc[:, "state"] = zillow_regions_new_pandas.apply(
    lambda x: check_state_in_str(x["region"]),
    axis=1)  # find state in the region column, store it in state column

zillow_regions_new_pandas.loc[:, "county"] = zillow_regions_new_pandas.apply(
    lambda x: check_county_in_str(x["region"]),
    axis=1)  # find county in the region column, store it in county column

# get city down below using zip --> city calculation?
zillow_regions_new_pandas.loc[:, "city"] = zillow_regions_new_pandas.apply(
    lambda x: check_city_in_str(x["region"]),
    axis=1)  # find city in the region column, store it in city column

zillow_regions_new_pandas.loc[:, "metro"] = zillow_regions_new_pandas.apply(
    lambda x: check_metro_in_str(x["region"]),
    axis=1)  # find metro in the regions column, store it in metro column

# convert cleaned regions pandas dataframe back to a pyspark dataframe
zillow_regions_new_pandas_to_pyspark = spark.createDataFrame(
    zillow_regions_new_pandas)
zillow_regions_new_pandas_to_pyspark.show(1)

# %% Update state, city, county where it's blank using zip code
# Where state or city or county is blank, add it in using the uszips.csv table from https://simplemaps.com/data/us-zips
# count how many rows where state is blank
zillow_regions_new_pandas_to_pyspark.createOrReplaceTempView(
    "zillow_regions_new_pandas_to_pyspark_temp")
# 2166 states are null
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state IS NULL"
).show()
zillow_regions_new_pandas_to_pyspark.filter(
    "city IS NULL").count()  # 3867 rows where city is null

# read in the zip to state file
zip_to_state = spark.read.csv("uszips.csv", header=True,
                              inferSchema=True).withColumnRenamed(
                                  "city", "city2")

# read in the zip to metro file - https://planiverse.wordpress.com/2019/01/18/mapping-zip-codes-to-msas-and-cbsas/
zip_to_metro = (spark.read.csv("zip_to_metro.csv",
                               header=True,
                               inferSchema=True).withColumnRenamed(
                                   "ZIP", "ZIP_metro").withColumnRenamed(
                                       "Metro", "Metro_metro"))

# replace "[not in a CBSA]" value with null
zip_to_metro = zip_to_metro.withColumn(
    "Metro_metro",
    when(col("Metro_metro") == "[not in a CBSA]",
         None).otherwise(col("Metro_metro")),
)

# join old table and new table, connecting by zip code where we choose left join because the values in zillow_regions_new_pandas_to_pyspark need to remain, regardless if zip is found in zip_to_state
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.join(
    zip_to_state,
    zillow_regions_new_pandas_to_pyspark.zip_code == zip_to_state.zip,
    "left",
)

# join old table and new table, connecting by zip code where we choose left join because the values in zillow_regions_new_pandas_to_pyspark need to remain, regardless if zip is found in zip_to_metro
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.join(
    zip_to_metro,
    zillow_regions_new_pandas_to_pyspark.zip_code == zip_to_metro.ZIP_metro,
    "left",
)

# count how many times state and state_id are null, different, prechecks
zillow_regions_new_pandas_to_pyspark.createOrReplaceTempView(
    "zillow_regions_new_pandas_to_pyspark_temp")
# 35451 rows
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state == state_id"
).show()
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state <> state_id"
).show()  # 50 rows
# 2166 rows
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state IS NULL"
).show()
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state_id IS NULL"
).show()  # 425 rows
spark.sql(
    "SELECT zip_code, zip, state, state_id FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state_id IS NULL"
).show()  # 425 rows

# count how many times city and city2 are null, different, prechecks
zillow_regions_new_pandas_to_pyspark.filter("city <> city2").count()  # 5681
zillow_regions_new_pandas_to_pyspark.filter("city IS NULL").count()  # 3867
zillow_regions_new_pandas_to_pyspark.filter("city2 IS NULL").count()  # 425

# count how many times county and county_name are null, different, prechecks
zillow_regions_new_pandas_to_pyspark.filter(
    "county <> county_name").count()  # 27995
zillow_regions_new_pandas_to_pyspark.filter("county IS NULL").count()  # 2984
zillow_regions_new_pandas_to_pyspark.filter(
    "county_name IS NULL").count()  # 425

# replace null state with state_id value, null city with city2 value, null county with county_name value, null metro values with Metro_metro
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.withColumn(
    "state",
    coalesce(
        zillow_regions_new_pandas_to_pyspark.state,
        zillow_regions_new_pandas_to_pyspark.state_id,
    ),
)
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.withColumn(
    "city",
    coalesce(
        zillow_regions_new_pandas_to_pyspark.city,
        zillow_regions_new_pandas_to_pyspark.city2,
    ),
)
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.withColumn(
    "county",
    coalesce(
        zillow_regions_new_pandas_to_pyspark.county,
        zillow_regions_new_pandas_to_pyspark.county_name,
    ),
)
zillow_regions_new_pandas_to_pyspark = zillow_regions_new_pandas_to_pyspark.withColumn(
    "metro",
    coalesce(
        zillow_regions_new_pandas_to_pyspark.metro,
        zillow_regions_new_pandas_to_pyspark.Metro_metro,
    ),
)

# count how many times state and state_id are null, different, postchecks. Now state column is 100% populated.
zillow_regions_new_pandas_to_pyspark.createOrReplaceTempView(
    "zillow_regions_new_pandas_to_pyspark_temp")
# 30712 rows
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state == state_id"
).show()
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state <> state_id"
).show()  # 52 rows
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state IS NULL"
).show()  # 0 rows
spark.sql(
    "SELECT count(*) FROM zillow_regions_new_pandas_to_pyspark_temp WHERE state_id IS NULL"
).show()  # 425 rows

# count how many times city and city2 are null, different, postchecks. now only 41 city rows are blank
zillow_regions_new_pandas_to_pyspark.filter("city <> city2").count()  # 6505
zillow_regions_new_pandas_to_pyspark.filter("city IS NULL").count()  # 41

# count how many times county and county_name are null, different, postchecks
zillow_regions_new_pandas_to_pyspark.filter(
    "county <> county_name").count()  # 34k
zillow_regions_new_pandas_to_pyspark.filter("county IS NULL").count()  # 5

# count how many times metro and Metro_metro are null, different, postchecks
zillow_regions_new_pandas_to_pyspark.filter(
    "metro <> Metro_metro").count()  # 0
zillow_regions_new_pandas_to_pyspark.filter("metro IS NULL").count()  # 11682
zillow_regions_new_pandas_to_pyspark.filter(
    "Metro_metro IS NOT NULL").count()  # 26485

# remove the unnecessary, duplicate columns
zillow_regions_new_pandas_to_pyspark_condensed = (
    zillow_regions_new_pandas_to_pyspark.drop(
        "region_type",
        "region",
        "region_str_len",
        "zip",
        "lat",
        "lng",
        "city2",
        "state_id",
        "state_name",
        "zcta",
        "parent_zcta",
        "county_fips",
        "county_name",
        "county_weights",
        "county_names_all",
        "county_fips_all",
        "imprecise",
        "military",
        "timezone",
        "ZIP_metro",
        "Metro_metro",
    ))

# ps.reset_option('compute.ops_on_diff_frames')

# %% Merge and clean all 3 tables for future analysis

# below code keeps causing python not found errors, so forcing exit so above can run with the below running
# https://itsmycode.com/python-was-not-found-run-without-arguments-to-install-from-the-microsoft-store-or-disable-this-shortcut-from-settings-manage-app-execution-aliases/
# raise KeyboardInterrupt

# merge tables, using primary keys: https://data.nasdaq.com/databases/ZILLOW/documentation
# use inner join for both because I only care where the e.g. data's region and region's region are both available, and e.g. the data's indicator and indicator's indicator are both available
# split into 2 separate joins, since doing all 3 at once leading to the above python was not found error
zillow_temp = zillow_data.join(zillow_regions_new_pandas_to_pyspark_condensed,
                               ["region_id"])
# zillow_temp.show(1)

try:
    zillow_all = zillow_temp.join(zillow_indicators_pyspark, ["indicator_id"])
    zillow_all.show(1)
except:
    print(
        "Likely Error - Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases. Or java.io.IOException: Cannot run program python3: CreateProcess error=3, The system cannot find the path specified"
    )

# count rentals...621601 rentals vs. 57611750 home values
zillow_all.groupby("category").count().show()

# keep only rentals for our analysis...if we need to switch it back to using non-rentals, can do <>
zillow_all_house = zillow_all.filter(zillow_all.category == "Home values")
zillow_all_rental = zillow_all.filter(zillow_all.category == "Rentals")

# sort by date:
zllow_all = zillow_all.sort("date", ascending=True)
zillow_all_house = zillow_all_house.sort("date", ascending=True)
zillow_all_rental = zillow_all_rental.sort("date", ascending=True)

# there are a lot of suffolk county properties with very high values, seem like outliers
# zillow_all_rental.sort('value', ascending = False).show(3)

# ps.reset_option('compute.ops_on_diff_frames')
# reduce size of zillow_all and zillow_house so that it can be written to csv
zillow_all_rental.count(
)  # 621601 see count of rental, which corresponds to 112 MB csv
zillow_rental = zillow_all_rental

zillow_all.count()  # 58576520
# due to size of the data, just keep .1%, otherwise writing to csv was failing
zillow_both = zillow_all.sample(fraction=0.01)

zillow_all_house.count()  # 57611750
# due to size of the data, just keep .1%, otherwise writing to csv was failing
zillow_house = zillow_all_house.sample(fraction=0.01)

# sometimes running all 3 to_csv's leads to terminations - prioritize house and rental
# zillow_both.toPandas().to_csv('zillow_both_backup_10212022.csv', index=False)
# zillow_house.toPandas().to_csv('zillow_house_backup_12312022.csv', index=False)
zillow_rental.toPandas().to_csv("zillow_rental_backup_12312022.csv",
                                index=False)
# zillow_all_rental.coalesce(1).write.csv('C:\Users\timtu\PycharmProjects\PySpark\zillow_all_backup.csv')

raise KeyboardInterrupt

# %% More analysis in tableau
# TODO update tableau with zillow_all, then use category to filter between house prices and rental?
# TODO or just use the rental file, load that and use that, especially if full file is too big
# TODO more groupby's to see average home value by different categories

# %% Create a pipeline - use state and zip population density to predict value
# https://spark.apache.org/docs/latest/ml-pipeline.html

# If there are issues above, then start from here, reload zillow_all_backup...
warnings.filterwarnings("ignore")  # ignores warnings

# %% Imports and setup spark session

# Create SparkSession object using all available cores on this local computer
spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

zillow_all_rental = spark.read.csv("zillow_rental_backup_12312022.csv",
                                   header=True,
                                   inferSchema=True,
                                   nullValue="NA")

# %% Data cleanup, Preparation, Train/test split
# do some prechecks on missing data to allow later ml functionality to work
zillow_all_rental.select([
    count(when(isnull(c), c)).alias(c) for c in zillow_all_rental.columns
]).show()  # count null values for all columns
# too long...zillow_all_rental.select([count(when(isnull(c), c)).alias(c)/zillow_all_rental.count() for c in zillow_all_rental.columns]).show() #count % values for all columns

# drop rows with null values in state, density, value, metro as these are going to be used later. It's ok that metro has high number of nulls, don't want to delete those rows
zillow_all_drop_nulls = zillow_all_rental.na.drop(
    subset=["state", "density", "value"]).drop("metro")
zillow_all_drop_nulls.show(5)

# recount after dropping nulls
zillow_all_drop_nulls.select([
    count(when(isnull(c), c)).alias(c) for c in zillow_all_drop_nulls.columns
]).show()  # count null values for all columns
zillow_all_drop_nulls.show(5)

# Split into training and testing sets in a 80:20 ratio
zillow_train, zillow_test = zillow_all_drop_nulls.randomSplit([0.8, 0.2],
                                                              seed=17)

zillow_train.show(1)

# %% Setup Clean/Condensed ML pipeline - first part of based on MLEIP, second is condensed of the previous

# Import class for creating a pipeline

# Construct a pipeline of all indexers, encoders, assemblers, regressors
# based on MLEIP Chapter03 pipelines sparkmllib_pipeline.ipynb
# https://stackoverflow.com/questions/62551905/output-column-already-exists-error-when-fit-with-pipeline-pyspark
# TODO Possibly add other columns besides state, using above link, to get better predictions - added categorical and numerical variables
# define list for storage stages of pipeline
stages_lr = []

# define the transformation stages for the categorical columns...metro has too many nulls so leave out
categoricalColumns = [
    "indicator_id", "zip_code", "state", "county", "city", "category"
]
for categoricalCol in categoricalColumns:
    # category indexing with string indexer
    stringIndexer = StringIndexer(inputCol=categoricalCol,
                                  outputCol=categoricalCol +
                                  "Index").setHandleInvalid(
                                      "keep")  # keep is for unknown categories
    # Use onehotencoder to convert cat variables into binary sparseVectors
    encoder = OneHotEncoder(
        inputCols=[stringIndexer.getOutputCol()],
        outputCols=[categoricalCol + "classVec"],
    )
    # Add stages. These are not run here, will be run later
    stages_lr += [stringIndexer, encoder]

# define impute stage for the numerical columns
numericalColumns = ["population", "density"]
numericalColumnsImputed = [x + "_imputed" for x in numericalColumns]
imputer = Imputer(inputCols=numericalColumns,
                  outputCols=numericalColumnsImputed)
stages_lr += [imputer]

# define numerical assembler first for scaling
numericalAssembler = VectorAssembler(inputCols=numericalColumnsImputed,
                                     outputCol="numerical_cols_imputed")
stages_lr += [numericalAssembler]

# define the standard scaler stage for the numerical columns
scaler = StandardScaler(inputCol="numerical_cols_imputed",
                        outputCol="numerical_cols_imputed_scaled")
stages_lr += [scaler]  # already a list so no need for brackets

# Perform assembly stage to bring together features
assemblerInputs = [c + "classVec" for c in categoricalColumns
                   ] + ["numerical_cols_imputed_scaled"]
# features contains everything, one hot encoded and numerical
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages_lr += [assembler]

# define the model stage at the end of the pipeline
lr = LinearRegression(labelCol="value", featuresCol="features", maxIter=10)
stages_lr += [lr]

# train Linear Regression model and make predictions
linreg_Pipeline_fit = Pipeline().setStages(stages_lr).fit(zillow_train)
linreg_Pipeline_predictions = linreg_Pipeline_fit.transform(zillow_test)
linreg_Pipeline_predictions.show(1)

# Generate predictions on testing data using the best model then calculate RMSE
# define the evaluator_regression
evaluator_mleip_linreg = RegressionEvaluator(labelCol="value")
print("RMSE =", evaluator_mleip_linreg.evaluate(linreg_Pipeline_predictions))

# %% Random Forest
# define list for storage stages of pipeline
stages_rf = []

# define the transformation stages for the categorical columns
categoricalColumns = [
    "indicator_id", "zip_code", "state", "county", "city", "category"
]
for categoricalCol in categoricalColumns:
    # category indexing with string indexer
    stringIndexer = StringIndexer(inputCol=categoricalCol,
                                  outputCol=categoricalCol +
                                  "Index").setHandleInvalid(
                                      "keep")  # keep is for unknown categories
    # Use onehotencoder to convert cat variables into binary sparseVectors
    encoder = OneHotEncoder(
        inputCols=[stringIndexer.getOutputCol()],
        outputCols=[categoricalCol + "classVec"],
    )
    # Add stages. These are not run here, will be run later
    stages_rf += [stringIndexer, encoder]

# define impute stage for the numerical columns
stages_rf += [imputer]

# define numerical assembler first for scaling
stages_rf += [numericalAssembler]

# define the standard scaler stage for the numerical columns
stages_rf += [scaler]  # already a list so no need for brackets

# Perform assembly stage to bring together features
stages_rf += [assembler]

# define the rf model stage at the end of the pipeline
rf = RandomForestRegressor(labelCol="value", featuresCol="features")
stages_rf += [rf]

# train Random Forest model and make predictions
rf_Pipeline_fit = Pipeline().setStages(stages_rf).fit(zillow_train)
rf_Pipeline_predictions = rf_Pipeline_fit.transform(zillow_test)
rf_Pipeline_predictions.show(1)

# Generate predictions on testing data using the best model then calculate RMSE
# define the evaluator_regression
evaluator_mleip_rf = RegressionEvaluator(labelCol="value")
print("RMSE =", evaluator_mleip_rf.evaluate(rf_Pipeline_predictions))

# %% #Previous working condensed pipeline (with just state):
print("One Hot Encoding...")
indexer = StringIndexer(inputCol="state", outputCol="stateIndex")
# Create an instance of the one hot encoder
encoder = OneHotEncoder(inputCols=["stateIndex"], outputCols=["stateOHE"])

print("Vectorizing...")
# Create an assembler object, create vector of features, for now follow Datacamp idea and use these 2 columns
assembler = VectorAssembler(inputCols=["stateOHE", "density"],
                            outputCol="features")
regression_model = LinearRegression(labelCol="value")
# elnet_model = LinearRegression(labelCol='value', regParam=.5, elasticNetParam=.5)
# rf_model = RandomForestRegressor(labelCol='value', maxBins=52)
print("Pipeline...")
pipeline_regression = Pipeline(
    stages=[indexer, encoder, assembler, regression_model])

# fit the pipeline on the training data, seemingly can only train 1 model at a time, otherwise column prediction already exists error
print("Fit...")
pipelineModel_regression = pipeline_regression.fit(zillow_train)

# make predictions on train data
print("Transform Train Data...")
predictionModel_regression_train = pipelineModel_regression.transform(
    zillow_train)

# show some of the linreg predictions and other key values
predictionModel_regression_train.select("stateOHE", "features", "value",
                                        "prediction").show(5)

# make predictions on test data
print("Transform Test Data...")
predictionModel_regression_test = pipelineModel_regression.transform(
    zillow_test)

# show some of the linreg test predictions and other key values
predictionModel_regression_test.select("stateOHE", "features",
                                       "prediction").show(5)

# define the evaluator_regression
evaluator_regression = RegressionEvaluator(labelCol="value")

# %% Run Train/Validation split

# Create parameter grid
paramGrid = (ParamGridBuilder().addGrid(
    regression_model.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(
        regression_model.fitIntercept,
        [False, True]).addGrid(regression_model.elasticNetParam,
                               [0.0, 0.5, 1.0]).build())
print("Number of models to be tested: ", len(paramGrid))

# create train/validation split
tvs = TrainValidationSplit(
    estimator=pipeline_regression,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator_regression,
    # 80% of the data will be used for training, 20% for validation.
    trainRatio=0.8,
)

# Run TrainValidationSplit, and choose the best set of parameters.
tvs_model = tvs.fit(zillow_train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
tvs_model.transform(zillow_test).show(3)

# Get the best model from tvs
tvs_best_model = tvs_model.bestModel

# Look at the stages (list of components in the model pipeline) in the best model
print(tvs_best_model.stages)

# Get the parameters for the LinearRegression object in the best model
tvs_best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
tvs_predictions = tvs_best_model.transform(zillow_test)
print("RMSE =", evaluator_regression.evaluate(tvs_predictions))

# %% Run Cross-Validation

# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression_model.regParam,
                        [0.01, 0.1, 1.0, 10.0]).addGrid(
                            regression_model.elasticNetParam, [0.0, 0.5, 1.0])

# Build the parameter grid
params = params.build()
print("Number of models to be tested: ", len(params))

# Create 3-fold cross-validator for now
cv_model = CrossValidator(
    estimator=pipeline_regression,
    estimatorParamMaps=params,
    evaluator=evaluator_regression,
    numFolds=3,
)

# fit the model using cv_model
cv_model = cv_model.fit(zillow_train)

# Get the best model from cross validation
cv_best_model = cv_model.bestModel

# Look at the stages (list of components in the model pipeline) in the best model
print(cv_best_model.stages)

# Get the parameters for the LinearRegression object in the best model
cv_best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
cv_predictions = cv_best_model.transform(zillow_test)
print("RMSE =", evaluator_regression.evaluate(cv_predictions))

##################################################################################

# %%backup - above are the condensed/simplified transforms. below is the underlying work before we could condense it above.
# do string indexer, then one-hot encoding on state, then scale the population density, then do modeling, then IndextoString to retrieve the original labels
# String Indexer
indexer = StringIndexer(inputCol="state", outputCol="stateIndex")
# identifies categories in the data, then create a new column with numeric index values
zillow_indexed = indexer.fit(zillow_all_drop_nulls).transform(
    zillow_all_drop_nulls)
# zillow_indexed.show()

# One-hot encoding
# Create an instance of the one hot encoder
encoder = OneHotEncoder(inputCols=["stateIndex"], outputCols=["stateDummy"])
# Apply the one hot encoder to the zillow data
model = encoder.fit(zillow_indexed)
# Apply the one hot encoder to the zillow data
zillow_encoded = model.transform(zillow_indexed)
zillow_encoded.select("state", "stateIndex",
                      "stateDummy").sort("stateIndex").show(5)

# scale the density, robustscaler to be robust against outliers, has issues with illegal argument exception, try later..maybe due to null values?
# https://stackoverflow.com/questions/61056160/illegalargumentexception-column-must-be-of-type-structtypetinyint-sizeint-in
# https://stackoverflow.com/questions/55536970/failed-to-execute-user-defined-functionvectorassembler
try:
    zillow_encoded.dtypes  # density is currently double, have to make it int
    zillow_encoded = zillow_encoded.withColumn(
        "density", zillow_encoded.density.cast("int"))
    zillow_encoded.printSchema(
    )  # density is currently double, have to make it int

    from pyspark.ml.feature import RobustScaler

    scaler = RobustScaler(
        inputCol="density",
        outputCol="scaledDensity",
        withScaling=True,
        withCentering=False,
        lower=0.25,
        upper=0.75,
    )
    # Compute summary statistics by fitting the RobustScaler
    scalerModel = scaler.fit(zillow_encoded)
except Exception:
    print("check above SO link, come back later")

# assemble feature vector
zillow_encoded.show(1)
# Create an assembler object, create vector of features, for now follow Datacamp idea and use these 2 columns
assembler = VectorAssembler(inputCols=["stateIndex", "density"],
                            outputCol="features")
zillow_assembled = assembler.transform(
    zillow_encoded)  # Consolidate predictor columns
zillow_assembled.select("stateIndex", "density", "value", "features").show(3)
zillow_assembled.count()

# %% ## Prepare data for ml
# reduce the data to just the columns we'll use for prediction
zillow_condensed = zillow_assembled.select("state", "stateIndex", "density",
                                           "value", "features")

# Split into training and testing sets in a 80:20 ratio
zillow_train, zillow_test = zillow_condensed.randomSplit([0.8, 0.2], seed=17)
zillow_train.show(1)
# Check that training set has around 80% of records
training_ratio = zillow_train.count() / zillow_condensed.count()
print(training_ratio)

# %% Linear Regression

# Create a regression object and train on training data
regression_model = LinearRegression(labelCol="value",
                                    regParam=0.3).fit(zillow_train)

# Summarize the model over the training set and print out some metrics
trainingSummary = regression_model.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# Create predictions for the testing data
linreg_predictions = regression.transform(zillow_test)

# Calculate the RMSE on testing data
linreg_rmse = RegressionEvaluator(
    labelCol="value").evaluate(linreg_predictions)
print("The linear regression test RMSE is", linreg_rmse)

# %% Regularization - Elastic Net

# Fit Lasso model (λ = .5, α = .5) to training data
elnet_regression = LinearRegression(labelCol="value",
                                    regParam=0.5,
                                    elasticNetParam=0.5).fit(zillow_train)

# Calculate the RMSE on testing data
elnet_rmse = RegressionEvaluator(labelCol="value").evaluate(
    elnet_regression.transform(zillow_test))
print("The elastic net test RMSE is", elnet_rmse)

# Look at the model coefficients
elnet_coeffs = elnet_regression.coefficients
print(elnet_coeffs)

# %% Random Forest Regression

# Train a RandomForest model.
rf_model = RandomForestRegressor(labelCol="value",
                                 maxBins=52).fit(zillow_train)

# Make predictions on test data.
rf_predictions = rf_model.transform(zillow_test)

# Select example rows to display.
rf_predictions.show(5)

# Calculate the RMSE on testing data
rf_rmse = RegressionEvaluator(labelCol="value").evaluate(rf_predictions)
print("The random forest test RMSE is", rf_rmse)

# %% K-means Clustering

# Trains a k-means model with k clusters.
kmeans = KMeans().setK(3).setSeed(1)
kmeans_model = kmeans.fit(zillow_train)

# Make predictions
kmeans_predictions = kmeans_model.transform(zillow_train)

# Evaluate clustering by computing Silhouette score
kmeans_evaluator = ClusteringEvaluator()

kmeans_silhouette = kmeans_evaluator.evaluate(kmeans_predictions)
print("Silhouette with squared euclidean distance = " + str(kmeans_silhouette))

# Shows the result.
centers = kmeans_model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# %% Create Pipeline

# Import class for creating a pipeline

# Construct a pipeline of all indexers, encoders, assemblers, regressors
indexer = StringIndexer(inputCol="state", outputCol="stateIndex")
# Create an instance of the one hot encoder
encoder = OneHotEncoder(inputCols=["stateIndex"], outputCols=["stateDummy"])
# Create an assembler object, create vector of features, for now follow Datacamp idea and use these 2 columns
assembler = VectorAssembler(inputCols=["stateIndex", "density"],
                            outputCol="features")
regression_model = LinearRegression(labelCol="value", regParam=0.3)
elnet_regression = LinearRegression(labelCol="value",
                                    regParam=0.5,
                                    elasticNetParam=0.5)
rf_model = RandomForestRegressor(labelCol="value", maxBins=52)
pipeline = Pipeline(stages=[
    indexer, encoder, assembler, regression_model, elnet_regression, rf_model
])

# fit the pipeline on the training data
model = pipeline.fit(zillow_train)

# make predictions on test data
prediction = model.transform(zillow_test)

# Train the pipeline on the training data, removing unnecessary columns which will be recreated by the pipeline
zillow_train.columns
zillow_train = zillow_train.drop("stateIndex", "features")
zillow_train.columns
zillow_test.columns
zillow_test = zillow_test.drop("stateIndex", "features")
zillow_test.columns

# error, saying state does not exist...add it back in when zillow_train is created
pipeline = pipeline.fit(zillow_train)

# Make predictions on the testing data
predictions = pipeline.transform(zillow_test)

#################################################################################

# %% Outlier detection via DBSCAN
# TODO time series anomaly detection to see if house prices are out of ordinary? https://towardsdatascience.com/anomaly-detection-with-time-series-forecasting-c34c6d04b24a
# Outlier detection - based on MLEIP Chapter01 - batch-anomaly
plt.rcParams.update({"font.size": 22})
rs = RandomState(MT19937(SeedSequence(123456789)))


def cluster_and_label(X, create_and_show_plot=True):
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)

    # Find labels from the clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #      % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #      % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f" %
          metrics.silhouette_score(X, labels))

    run_metadata = {
        "nClusters": n_clusters_,
        "nNoise": n_noise_,
        "silhouetteCoefficient": metrics.silhouette_score(X, labels),
        "labels": labels,
    }
    if create_and_show_plot == True:
        fig = plt.figure(figsize=(10, 10))
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [
            plt.cm.cool(each) for each in np.linspace(0, 1, len(unique_labels))
        ]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = labels == k
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "^",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

        plt.xlabel("Standard Scaled Ride Dist.")
        plt.ylabel("Standard Scaled Ride Time")
        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()
    else:
        pass
    return run_metadata


# Run above function on value and density fields, create some nice plots, runs for a few minutes
dbscan_df = zillow_all_rental[["value", "density"]].toPandas().dropna()
dbscan_df.isnull().sum()
dbscan_results = cluster_and_label(dbscan_df)  # plots don't show up in pycharm
# zillow_all_rental['label'] = dbscan_results['labels']

# %% Prophet Time Series
# Prophet Time Series prediction using MLEIP Chapter01 forecasting-api electricity-demand-forecast.ipynb and item-demand-forecast.ipynb
plt.rcParams.update({"font.size": 22})
# pip install prophet

# create date field to Date format
zillow_all_drop_nulls_date = zillow_all_drop_nulls.select(
    "*",
    to_date(col("date"), "YYYY-MM-DD").alias(
        "DateOfSale"))  # .toPandas().astype  #plot(x='date', y = 'value')

# doesn't work yet... https://spark.apache.org/docs/3.2.1/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.plot.line.html
zillow_all_drop_nulls_date.plot.line(x="DateOfSale", y="value")

# functions prep for prophet
seasonality = {"yearly": True, "weekly": False, "daily": False}


def time_split_train_test(df, time_series_splits, seasonality=seasonality):
    # for outputting
    df_results = pd.DataFrame()

    for i, (train_i, test_i) in enumerate(time_series_splits.split(df)):
        # grab split data
        df_train = df.copy().iloc[train_i, :]
        df_test = df.copy().iloc[test_i, :]

        # create Prophet model
        model = Prophet(
            yearly_seasonality=seasonality["yearly"],
            weekly_seasonality=seasonality["weekly"],
            daily_seasonality=seasonality["daily"],
        )

        # train and predict
        model.fit(df_train)
        predicted = model.predict(df_test)

        # combine pred and training df's for plotting
        df_pred = predicted.loc[:, ["ds", "yhat"]]

        df_pred["y"] = df_test["y"].tolist()

        # Train or Test?
        df_train["train"] = True
        df_pred["train"] = False

        df_sub = df_train.append(df_pred).reset_index(drop=True)
        df_sub["split"] = i
        # calculating rmse for the split
        df_sub["rmse"] = (np.mean((df_sub.yhat - df_sub.y)**2))**0.5

        df_results = df_results.append(df_sub).reset_index(drop=True)
    return df_results


def forecaster_train_and_export(df, seasonality):
    # create Prophet model
    model = Prophet(
        yearly_seasonality=seasonality["yearly"],
        weekly_seasonality=seasonality["weekly"],
        daily_seasonality=seasonality["daily"],
    )

    # train and predict
    model.fit(df)
    return model


# %% Time series regression to predict next year's prices - preparing for model running
sns.set()

# set the font sizes of the plots:
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot the median price vs. date data
zillow_all_rental.value = pd.to_numeric(
    zillow_all_rental["value"])  # convert value to numeric
zillow_all_rental.date = pd.to_datetime(
    zillow_all_rental["date"], format="%Y-%m-%d")  # convert date to date
zillow_all_rental.region_id = zillow_all_rental["region_id"].astype(str)
zillow_price_by_date = zillow_all_rental.groupby(
    ["date"], as_index=False).median(["value"])  # this is our time series data
zillow_price_by_date.info()
zillow_price_by_date.describe().T

# plot data
# plt.rcParams['font.size'] = '24'
plt.ylabel("Median Price")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.plot(zillow_price_by_date["date"], zillow_price_by_date["value"])
plt.show(block=True)

# check if data is stationary by Augmented Dickey Fuller test
adfuller_test = adfuller(zillow_price_by_date.value.dropna())
# p-value = .99, cannot reject Ho that time series is non-stationary
print("p-value: ", adfuller_test[1])

# plot differencing, autocorrelation - first differencing provides stationarity
plt.rcParams.update({"figure.figsize": (9, 7), "figure.dpi": 120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(zillow_price_by_date.value)
axes[0, 0].set_title("Original Series")
plot_acf(zillow_price_by_date.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(zillow_price_by_date.value.diff())
axes[1, 0].set_title("1st Order Differencing")
plot_acf(zillow_price_by_date.value.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(zillow_price_by_date.value.diff().diff())
axes[2, 0].set_title("2nd Order Differencing")
plot_acf(zillow_price_by_date.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# as per above, stationarity reached with 1 differencing, tentatively fix order of differencing as 1
# Adf test
ndiffs(zillow_price_by_date.value, test="adf")  # 1
# KPSS test
ndiffs(zillow_price_by_date.value, test="kpss")  # 1
# PP test
ndiffs(zillow_price_by_date.value, test="pp")  # 0
# (1,1,0)

# PACF plot of 1st differenced series
plt.rcParams.update({"figure.figsize": (9, 3), "figure.dpi": 120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(zillow_price_by_date.value.diff())
axes[0].set_title("1st Differencing")
axes[1].set(ylim=(0, 5))
plot_pacf(zillow_price_by_date.value.diff().dropna(), ax=axes[1])
plt.show()

# ACF plot of 1st differenced series
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(zillow_price_by_date.value.diff())
axes[0].set_title("1st Differencing")
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
residuals.plot(kind="kde", title="Density", ax=ax[1])
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
plt.plot(train, label="training")
plt.plot(test, label="actual")
plt.plot(fc_series, label="forecast")
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=8)
plt.show()

# use auto-arima to determine lowest AIC model

auto_arima_model = pm.auto_arima(
    zillow_price_by_date.value,
    start_p=1,
    start_q=1,
    test="adf",  # use adftest to find optimal 'd'
    max_p=3,
    max_q=3,  # maximum p and q
    m=1,  # frequency of series
    d=None,  # let model determine 'd'
    seasonal=False,  # No Seasonality
    start_P=0,
    D=0,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

print(auto_arima_model.summary())

# plot residuals - overall looks ok
auto_arima_model.plot_diagnostics(figsize=(7, 5))
plt.show()

# forecast using optimal model
# Forecast
n_periods = 24
# make predictions for the next 24 periods
fc, confint = auto_arima_model.predict(n_periods=n_periods,
                                       return_conf_int=True)
index_of_fc = np.arange(len(zillow_price_by_date.value),
                        len(zillow_price_by_date.value) +
                        n_periods)  # store index values of forecasted data

# make series for plotting purpose
# turn fc array into fc_series pandas series
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(zillow_price_by_date.value)
plt.plot(fc_series, color="darkgreen")
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color="k",
                 alpha=0.15)

plt.title("Final Forecast of WWW Usage")
plt.show()

# Build SARIMA model to capture seasonal differences

# Seasonal - fit stepwise auto-ARIMA
zillow_price_by_date = zillow_price_by_date.set_index(
    "date")  # make the date an index for sarima to work

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(zillow_price_by_date[:], label="Original Series")
axes[0].plot(zillow_price_by_date[:].diff(1), label="Usual Differencing")
axes[0].set_title("Usual Differencing")
axes[0].legend(loc="upper left", fontsize=10)

# Seasonal Differencing
axes[1].plot(zillow_price_by_date[:], label="Original Series")
axes[1].plot(zillow_price_by_date[:].diff(12),
             label="Seasonal Differencing",
             color="green")
axes[1].set_title("Seasonal Differencing")
plt.legend(loc="upper left", fontsize=10)
plt.suptitle("House Prices", fontsize=16)
plt.show()

# fit the sarima model
auto_sarima_model = pm.auto_arima(
    zillow_price_by_date,
    start_p=1,
    start_q=1,
    test="adf",
    max_p=3,
    max_q=3,
    m=12,
    start_P=0,
    seasonal=True,
    d=None,
    D=1,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

# show summary - best model was SARIMAX(1, 1, 2)x(0, 1, [1], 12)
auto_sarima_model.summary()

# Forecast next 24 months
n_periods = 24
fitted, confint = auto_sarima_model.predict(n_periods=n_periods,
                                            return_conf_int=True)
index_of_fc = pd.date_range(zillow_price_by_date.index[-1],
                            periods=n_periods,
                            freq="MS")

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(zillow_price_by_date)
plt.plot(fitted_series, color="darkgreen")
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color="k",
                 alpha=0.15)

plt.title("SARIMA - Final Forecast of Housing Prices")
plt.show()

# use skforecast to forecast data...https://www.cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html
# Data manipulation
# ==============================================================================

# Plots
# ==============================================================================
plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 1.5

# Modeling and Forecasting
# ==============================================================================

# Warnings configuration
# ==============================================================================
# warnings.filterwarnings('ignore')

# set date as index if not already, sort by date
zillow_price_by_date = zillow_price_by_date.set_index("date")
zillow_price_by_date = zillow_price_by_date.sort_index()

# Split data into train-test
# ==============================================================================
steps = 36
data_train = zillow_price_by_date[:-steps]  # train is all data up to n-36
data_test = zillow_price_by_date[-steps:]  # test is the final 36

print(
    f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})"
)
print(
    f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})"
)

fig, ax = plt.subplots(figsize=(9, 4))
data_train["value"].plot(ax=ax, label="train")
data_test["value"].plot(ax=ax, label="test")
ax.legend()
plt.show()

# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123), lags=6)

forecaster.fit(y=data_train["value"])
forecaster

# Predictions...have to do some manipulation to include dates, give proper format
# ==============================================================================
steps = 36
predictions = pd.DataFrame([forecaster.predict(steps=steps)[:steps]]).T
len(predictions) == len(data_test.index)
predictions["date"] = data_test.index
len(predictions) == len(data_test)
predictions = predictions.set_index("date")
predictions.head(5)

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train["value"].plot(ax=ax, label="train")
data_test["value"].plot(ax=ax, label="test")
predictions.plot(ax=ax, label="predictions")
ax.legend()
plt.show()

# Hyperparameter Grid search
# ==============================================================================
steps = 36
forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=12,  # This value will be replaced in the grid search
)

# Lags used as predictors
lags_grid = [10, 20]

# Regressor's hyperparameters
param_grid = {"n_estimators": [100, 500], "max_depth": [3, 5, 10]}

results_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=data_train["value"],
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=steps,
    refit=True,
    metric="mean_squared_error",
    initial_train_size=int(len(data_train) * 0.5),
    fixed_train_size=False,
    return_best=True,
    verbose=False,
)

results_grid

# Predictions
# ==============================================================================
predictions = pd.DataFrame([forecaster.predict(steps=steps)[:steps]]).T
len(predictions) == len(data_test.index)
predictions["date"] = data_test.index
len(predictions) == len(data_test)
predictions = predictions.set_index("date")
predictions.head(5)

# Test error
# ==============================================================================
error_mse = mean_squared_error(y_true=data_test["value"], y_pred=predictions)

print(f"Test error (mse): {error_mse}")

# split price data into train (pre-2020) and test (post-2020)
train = zillow_price_by_date[zillow_price_by_date.date < pd.to_datetime(
    "2020-01-01", format="%Y-%m-%d")]
test = zillow_price_by_date[zillow_price_by_date.date >= pd.to_datetime(
    "2020-01-01", format="%Y-%m-%d")]

# plot the train and test data...for some reason dates are numeric instead of dates...
# TODO figure out why x-axis is numbers instead of dates --> was using index instead of date, had to convert the date to an index
plt.plot(train, color="black")
plt.plot(test, color="red")
plt.ylabel("House Price", fontsize=20)
plt.xlabel("Date", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.title("Train/Test split for Housing Data", fontsize=24)
plt.show()

# define y as the train data's housing prices
y = train["value"]

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
y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0],
                                             end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color="green", label="Predictions")
plt.legend()
plt.show()

# calculate RMSE
arma_rmse = np.sqrt(
    mean_squared_error(test["value"].values, y_pred_df["Predictions"]))
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
y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0],
                                              end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color="Yellow", label="ARIMA Predictions")
plt.legend()
plt.show()

# calculate RMSE
arima_rmse = np.sqrt(
    mean_squared_error(test["value"].values, y_pred_df["Predictions"]))
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
y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0],
                                                end=y_pred_df.index[-1])
y_pred_df.index = test.index  # set test index to y pred index...
y_pred_out = y_pred_df["Predictions"]  # store the predictions
# plot results
plt.plot(y_pred_out, color="Blue", label="SARIMAX Predictions")
plt.legend()
plt.show()

# calculate RMSE
sarimax_rmse = np.sqrt(
    mean_squared_error(test["value"].values, y_pred_df["Predictions"]))
print("RMSE: ", sarimax_rmse)
