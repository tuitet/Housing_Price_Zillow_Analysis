# Rental Price Zillow Analysis

## Summary
**Goal**: The goal of this project is to analyze geographical and temporal trends of US rental real estate prices since 2014. This includes visualizing rental data over time by region using Tableau. <!-- , and performing forecasting and anomaly detection to analyze expected future prices and detect unusual house prices. -->

**Dataset**: Popular housing platform Zillow stores housing price data down to the zip code level. Quandl collects this data and stores the time-series data in 3 tables:  
- Zillow Data: Stores housing price data over time and region.
- Zillow Indicators: Stores information on the type of property<!-- (for this analysis, we focused on single family homes) -->. 
- Zillow Regions: Stores region reference data.
More details: https://data.nasdaq.com/databases/ZILLOW/documentation

**Approach**: We use API's, webscraping, and data cleaning and merging to prepare a clean dataset from the original messy, disparate data. After collecting and pre-processing the data, we use the cleaned dataset to perform 2 main actions:
- Visualize Rental Prices: We use Tableau to visualize rental price changes over time and across the US.
- Predict Rental Prices: We use PySpark's MLLib regression functions to predict rental prices based on other available data.  
<!-- - Forecast future housing prices: We use several time-series approaches (... ) to analyze the existing data (including detrend and remove seasonality) and forecast house prices for the following year. 
- Detect anomalous house prices (by region?): We use ___ anomaly detection approaches to identify housing prices that are distinctly different than the rest... -->
The main goal is the visualization. Rental price prediction is more for practice in using the MLLib library. 

**Results**: 
- From the visualization, we saw as expected that the Northeast and California consistently had the highest median rental prices, while other expected metro regions also showed large increases over time (especially in Colorado, Washington, Texas, and Florida). We also see that up to 2021, rental prices across the country were slowly increasing. But in 2022, rental price increases accelerated, aligned with the headlines around inflation. More details: <!-- Also the visualization reminded me of the 2008 housing crisis, as we see some run-up of housing prices before 2008, but some drop-off after 2008. It seems like middle America was more negatively impacted than the coasts by the recession. Maybe add more... ... --> https://public.tableau.com/app/profile/tim1014/viz/ZillowRentalPrices-2014-2022/MedianRentalPricePerQuarterbyMetro
- From the linear and random forest regressions, we see relatively large RMSE values (around $340 for linear regression, $700 for random forest). However, we achieved the goal of learning how to setup an ML pipeline in PySpark with many of the common stages (string indexer + one hot encoder, imputer, vector assembler, scaler, and the regression) and apply it to fit the training data and measure its performance on the test data.


## Key skills being practiced in this project
There were several skills I wanted to explore, and thus why I chose this dataset. This project tested several
key data mining and statistical learning skills, including:  
- **Data Collection via API's and web scraping**: I wanted to learn how to setup a data pipeline using API's to collect frequently-updated data from a website (Quandl) and practice my webscraping skills to collect the metropolitan region data from Wikipedia.   
- **Data Cleaning and Wrangling**: I wanted to improve my data cleansing and data wrangling skills. This dataset was across 3 tables and had some messy columns, so this gave me an opportunity to practice cleaning and merging data into a single dataset which I could use later for visualization and analysis. 
For example, the region table's region column was just a semicolon-separated list of whichever region data was available. Some data points had just 1 piece of region information (the zipcode), while others included up to 5 pieces of region information (zipcode, city, county, state, metro area). Therefore separating the region column into separate columns took time.  
- **Data Visualization**: I wanted to practice creating clear and useful visualizations using a BI tool like Tableau. After cleaning the data, I was able to visualize the change in median rental price from 2014-2022, and region-level rental sales and prices. 

<!-- - **Build Time-Series Models for detrending and forecasting**: ...  
- **Build anomaly-detection models to identify unusually-high house prices**: 
- **Written Communication**: I wanted to practice my written communication skills, presenting technical data mining results to
a non-technical audience.  -->

## To-do:
Below are potential future actions that could be done:  
- Update time series to use prophet
- Perform anomaly detection
- Update Tableau file with full dataset
- Improve MLLib models (trying different types of scaling or models, selecting a subset of variables)
- Possibly filter by region or by single family homes or rentals and analyze in more detail
