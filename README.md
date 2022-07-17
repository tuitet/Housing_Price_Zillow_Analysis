# Housing Price Zillow Analysis

## Summary
**Goal**: The goal of this project was to analyze geographical and temporal trends of US housing prices since 1996. This includes visualizing housing data over time by region using Tableau, and performing forecasting and anomaly detection to analyze expected future prices and detect unusual house prices.

**Dataset**: Popular housing platform Zillow stores housing price data down to the zip code level. Quandl collects this data and stores the time-series data in 3 tables:  
- Zillow Data: Stores housing price data over time and region.
- Zillow Indicators: Stores information on the type of house (for this analysis, we focus on single family homes). 
- Zillow Regions: Stores region master data.

**Approach**:  classification methods can be used to predict outcomes based on observed data, and infer relationships between those observed features and the outcome. We will build 7 classification models that predict the variant (for the purpose of this analysis, whether it's the Original variant or Delta variant) and infer the relationship between the variant and the observed demographic and medical data.  

**Results**: The 7 models perform similarly, but Extreme Gradient Boosting performs the best with a prediction accuracy of 64.19% and an Area Under Curve (AUC) of 62.43%. For inference, age group (those younger than 20 are especially more likely to contract the Delta variant, while those older than 40 skew towards the Original variant) and pre-existing medical conditions (those with pre-existing medical conditions were more likely to contract the Delta variant, while those without pre-existing medical conditions were more likely to contract the Original variant) have the strongest relationship with type of variant.  

**Conclusions**: These machine learning classification models use observed medical and demographic data to predict the variant and infer relationships between the observed data and the variant. These predictions are better than random guessing and help understand who is more susceptible to certain variants. However, there are many false positives and false negatives that make it difficult to trust the results without more context. These analyses and results can be used as a tool for understanding variant spread in the aggregate and inform targeted public messaging, but would have to be one tool among many for guiding macro policy decisions.

## Key skills being practiced in this project
There were several skills I wanted to explore, and thus why I chose this dataset. This project tested several
key data mining and statistical learning skills, including:  
- **Data Collection via API's and web scraping**: I wanted to learn how to setup a data pipeline using API's to collect frequently-updated data from a website (Quandl) and practice my webscraping skills to collect the metropolitan region data from Wikipedia.   
- **Data Cleaning and Wrangling**: I wanted to improve my data cleansing and data wrangling skills. 
For example, the region table's region column was just a semicolon-separated list of whichever region data was available. Some data points had just the zipcode information, while others included data up to the metropolitan region. Therefore separating the region column into separate columns took time.  


- **Data Visualization**: I wanted to practice creating clear and useful visualizations. Several simple yet powerful visuals
helped quickly analyze the data and make decisions based on it.  
- **Build Classification Models**: I wanted to apply several classification models: Logistic Regression, Naïve Bayes, Random Forest,
Stochastic Gradient Boosting, Single-Layer Neural Network, Extreme
Gradient Boosting, Multilayer Perceptron Network.  
- **Hyperparameter Tuning**: I wanted to perform hyperparameter tuning on many different models, to understand the
diversity of hyperparameter options across models.  
- **Written Communication**: I wanted to practice my written communication skills, presenting technical data mining results to
a non-technical audience.  
