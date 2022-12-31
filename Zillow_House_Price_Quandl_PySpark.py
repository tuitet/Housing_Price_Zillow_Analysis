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
warnings.filterwarnings("ignore")  # ignores warnings

# TODO convert to jupyter notebook? https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html#ui

# %% Imports and setup spark session

# avoid later errors https://stackoverflow.com/questions/68705417/pycharm-error-java-io-ioexception-cannot-run-program-python3-createprocess
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
# Read data from downloaded CSV file from website
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

#get zillow indicator data, show first row
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

# %% Explore Zillow data

# explore pyspark data check options
zillow_indicators.isnull().values.any()  # check if any values are null 0
# check number of null values 0
zillow_regions.filter("region_id IS NULL").count()
# create a Table/View from the PySpark DataFrame to use sql queries on
zillow_data.createOrReplaceTempView("zillow_data_temp")
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

# keep only the regions which go down to the zip level, which is the most frequent
zillow_regions.createOrReplaceTempView("zillow_regions_temp")
zillow_regions_new = spark.sql(
    "SELECT * FROM zillow_regions_temp WHERE region_type = 'zip'")
# zillow_regions_new.show(1)
zillow_regions_new.count()  # 31204

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

# Extract the content from the region's column, note city column takes a while to generate, and returns false positives when metro name and city names overlap

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

# %% Merge and clean all 3 tables for future analysis

# merge tables, using primary keys: https://data.nasdaq.com/databases/ZILLOW/documentation
# use inner join for both because I only care where the e.g. data's region and region's region are both available, and e.g. the data's indicator and indicator's indicator are both available
# split into 2 separate joins, since doing all 3 at once leading to the above python was not found error
zillow_temp = zillow_data.join(zillow_regions_new_pandas_to_pyspark_condensed,
                               ["region_id"])

try:
    zillow_all = zillow_temp.join(zillow_indicators_pyspark, ["indicator_id"])
    zillow_all.show(1)
    # count rentals...621601 rentals vs. 57611750 home values
    zillow_all.groupby("category").count().show()
except:
    print(
        "Likely Error - Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases. Or java.io.IOException: Cannot run program python3: CreateProcess error=3, The system cannot find the path specified"
    )


# keep only rentals for our analysis...if we need to switch it back to using non-rentals, can do <>
zillow_all_house = zillow_all.filter(zillow_all.category == "Home values")
zillow_all_rental = zillow_all.filter(zillow_all.category == "Rentals")

# sort by date:
zllow_all = zillow_all.sort("date", ascending=True)
zillow_all_house = zillow_all_house.sort("date", ascending=True)
zillow_all_rental = zillow_all_rental.sort("date", ascending=True)

# reduce size of zillow_all and zillow_house so that it can be written to csv
zillow_all_rental.count()  # 621601 see count of rental, which corresponds to 112 MB csv
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

# TODO: See visualization in Tableau: https://public.tableau.com/app/profile/tim1014/viz/ZillowRentalPrices-2014-2022/MapofMedianRentalPricesbyMetro2014-20222?publish=yes


# %% Create a pipeline - use state and zip population density to predict value
# https://spark.apache.org/docs/latest/ml-pipeline.html

# %% Data cleanup, Preparation, Train/test split
# do some prechecks on missing data to allow later ml functionality to work
zillow_all_rental.select([
    count(when(isnull(c), c)).alias(c) for c in zillow_all_rental.columns
]).show()  # count null values for all columns

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

# %% Setup Clean/Condensed ML pipeline

# Construct a pipeline of all indexers, encoders, assemblers, regressors
# based on MLEIP Chapter03 pipelines sparkmllib_pipeline.ipynb

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