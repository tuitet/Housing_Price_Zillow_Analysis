# Housing Price Zillow Analysis

test1...

## Summary

**Goal**: The goal of this project is to analyze geographical and temporal trends of US housing prices since 1996. This includes visualizing housing data over time by region using Tableau, and performing forecasting and anomaly detection to analyze expected future prices and detect unusual house prices.

**Dataset**: Popular housing platform Zillow stores housing price data down to the zip code level. Quandl collects this data and stores the time-series data in 3 tables:

- Zillow Data: Stores housing price data over time and region.
- Zillow Indicators: Stores information on the type of house (for this analysis, we focused on single family homes).
- Zillow Regions: Stores region reference data.
  More details: https://data.nasdaq.com/databases/ZILLOW/documentation

**Approach**: We use API's, webscraping, and data cleaning and merging to prepare a clean dataset from the original messy, disparate data. After collecting and pre-processing the data, we use the cleaned dataset to perform 3 actions:

- Visualize Housing Prices: We use Tableau to visualize house price changes over time and across the country.
- Forecast future housing prices: We use several time-series approaches (... ) to analyze the existing data (including detrend and remove seasonality) and forecast house prices for the following year.
- Detect anomalous house prices (by region?): We use \_\_\_ anomaly detection approaches to identify housing prices that are distinctly different than the rest...
  The main goal is the visualization - forecasting and anomaly detection are more for practice...

**Results**:

- From visualization, we saw as expected that the Northeast and California consistently had the highest median housing prices, while there were some spikes at different points in time in Colorado, Washington, Texas, and Florida. Also the visualization reminded me of the 2008 housing crisis, as we see some run-up of housing prices before 2008, but some drop-off after 2008. It seems like middle America was more negatively impacted than the coasts by the recession. Maybe add more...
- From forecasting, inconclusive...indicate this was more for practice...?? or just leave out..?
- From anomaly detection...
- .........The 7 models perform similarly, but Extreme Gradient Boosting performs the best with a prediction accuracy of 64.19% and an Area Under Curve (AUC) of 62.43%. For inference, age group (those younger than 20 are especially more likely to contract the Delta variant, while those older than 40 skew towards the Original variant) and pre-existing medical conditions (those with pre-existing medical conditions were more likely to contract the Delta variant, while those without pre-existing medical conditions were more likely to contract the Original variant) have the strongest relationship with type of variant.

**Conclusions**: ............ These machine learning classification models use observed medical and demographic data to predict the variant and infer relationships between the observed data and the variant. These predictions are better than random guessing and help understand who is more susceptible to certain variants. However, there are many false positives and false negatives that make it difficult to trust the results without more context. These analyses and results can be used as a tool for understanding variant spread in the aggregate and inform targeted public messaging, but would have to be one tool among many for guiding macro policy decisions.

## Key skills being practiced in this project

There were several skills I wanted to explore, and thus why I chose this dataset. This project tested several
key data mining and statistical learning skills, including:

- **Data Collection via API's and web scraping**: I wanted to learn how to setup a data pipeline using API's to collect frequently-updated data from a website (Quandl) and practice my webscraping skills to collect the metropolitan region data from Wikipedia.
- **Data Cleaning and Wrangling**: I wanted to improve my data cleansing and data wrangling skills. This dataset was across 3 tables and had some messy columns, so this gave me an opportunity to practice cleaning and merging data into a single dataset which I could use later for visualization and analysis.
  For example, the region table's region column was just a semicolon-separated list of whichever region data was available. Some data points had just the zipcode information, while others included data up to the metropolitan region. Therefore separating the region column into separate columns took time.
- **Data Visualization**: I wanted to practice creating clear and useful visualizations using a BI tool like Tableau. After cleaning the data, I was able to visualize the change in median housing price from 1996-2022, and state-level housing sales and prices per state.

- **Build Time-Series Models for detrending and forecasting**: ...
- **Build anomaly-detection models to identify unusually-high house prices**:
- **Written Communication**: I wanted to practice my written communication skills, presenting technical data mining results to
  a non-technical audience.
