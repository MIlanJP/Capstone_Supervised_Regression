# Capstone_Supervised_Regression

## Seoul Bike Sharing Demand Prediction
# (Supervised Regression)
## Milan Gowda  J.P, Data science trainee,
1. Currently, Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the 
waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.
Seoul Bike Sharing Demand Prediction
(Supervised Regression)
Milan Gowda  J.P, Data science trainee,
Almabetter, Bangalore



Attributes:
Numerical:
1.	Date: Its format is in year-month-day 
2.	Hour: Hour of the day
3.	Temperature: In Celcius unit 
4.	Humidity: In percentage %
5.	Windspeed m/s:
6.	Visibility - 10m
7.	Dew point temperature – Celsius
8.	Solar radiation - MJ/m2
9.	Rainfall – mm
10.	Snowfall – cm
Categorical:
11.	Seasons - Winter, Spring, Summer, Autumn
12.	Holiday - Holiday/No holiday
13.	Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)
Target:
14.	Rented Bike Count: Count of Bike rented each hour
2.	Introduction
Bike Sharing Demand on an hourly basis is predicted by using tested specific effective machine learning models to avoid the waiting line by using features like weather conditions like temperature, humidity, Seasons on hour basis, with details about the type of the day whether it’s a non-functional or functional, Holiday or not, this is called Bike Sharing Demand Prediction, i.e. we need to predict the demand of the bike on every hour with precalculated features that collected related to the details about the weather on hour basis and details about the day
3.	Features effect on Target:
Weather conditions like rainy and winter season can affect the Bike Sharing Business because people during these seasons people opt for public transport or prefer to travel through car, If the visibility is less due to high risk of accidents people tend to not travel in bikes, So we can see that there is a negative correlation between bikes being rented and rainfall and snowfall. i.e. demand for bike rent decreases when there is rainfall and snowfall  

The temperature has a positive correlation with the Bike rent demand 
In simple terms, if the weather is good to go out, we can see a rise in the bike rent count.
4.	Loading Data set:
Once the data set is been loaded :
We check for the (1) head and (2)tail of the dataset (3)stats of the dataset (4) shape of the dataset (5) and information about the dataset that gives information about the number of non-null values on features with its datatype.
types: float64(6), int64(4), 
object(4) 
This is the information about the dataset with the number of features divided into datatypes.

4 a.  Extracting feature from the date 
The “date” is in the format DD/MM/YYYY which has information about the Day 
and Month whereas these Days and month
can give Machine learning Models some 
hints about the Bike rent demand 


5.	Checking Missing values:
Once the dataset is spied for the NAN 
Values, there is no such missing values 
Found. If no missing values are found, this condition does not mean that the dataset is ready for further process. Maybe missing 
values can be in the form of  ? (question
 mark) and 0(Zeros) we can enter a question mark or zero if the value is unknown.
Dataset was checked once again if there is any presence of? (question mark symbol)
Present and also for the presence of zero 
was checked When the zero value is 
present in the features treating them as missing values depend Upon the domain 
knowledge
  

We can see that zero is present in most 
all the rows taking the major portion of the data, Hence if we remove all of them 
we might face a huge loss of data
So we need to check if zero is commonly 
Occurred in the feature given.
As we see that features like Rain Fall, 
Snow Fall, Solar radiation, Windspeed, 
Temperature, Due point temperature and
Humidity can have its values as zero and 
It’s common in that region (Seoul) 

6.	Exploratory Data Analysis
6 a. Uni-variate analysis
Distributions of all the numerical columns have been plotted in histograms with boxplots
Total 3 plots are there:
1.	Histogram (Histogram 1)
2.	Boxplot
3.	Histogram concerning Seasons
		(Histogram 2)
1.	Temperature and Due point 
temperature
Temperature feature is been normally 
Distributed, box plot does not show 
Any outliers, Histogram concerned with 
Seasons shows that temperature in all 
seasons are normally distributed slightly 
overlapping on each other 
2. Humidity
Histogram 1:
•	There is a small trail towards the left (negligible).
•	There are no outliers in the data
(Boxplot)
•	There is a mix of Gaussians (Normal distribution of different seasons closely overlapping) But no outliers in the data. (Histogram 2)
3. Visibility
•	There are no outliers in the data
•	But the distributions are not normal
But Other Features Like:
•	Wind Speed
•	Solar Radiation
•	Rain Fall
•	Snow Fall
Does not follow a normal distribution and mainly these features are cursed with outliers where if we attempt to remove all the outliers we might lose almost 80% of the data provided
Because rainfall, Snowfall, Solar Radiation, and Wind speed always remains zero most of the time and they fluctuate when the season changes and occur only during some specific seasons
For example, Rainfall occurs only during the rainy season whereas it will not occur during summer, spring, winter so the distribution is more concentrated around the value zero so the data points collected during the rainy season might be treated as outliers and if we try to remove the outliers we might end up removing 25% if the genuine data.
And This happens in the case of Snow Fall and Solar Radiation and case if we remove them as outliers we’ll only end up with 10 to 15 percent of the data
6 b. Bi-variate Analysis 
 
Bikes rent Demand is increased during March and decreases during December. 
1.	These both Temperature and Due point temperature have a positive correlation with the target variable
2.	Rainfall and Snowfall have a negative correlation with the target variable
3.	Temperature and Due point temperature are highly correlated.
4.	Most commonly average bikes  rented are more in days that are not a holiday
5.	Most of the time maximum bikes rented are more during morning and evening time
6.	Bike rent Demand will start increasing from May 
7.	Seoul is a cold region because sometimes the average due point temperature during wintertime is around -17 degrees in Celcius
7.  Checking and Handling Outliers
From the univariate analysis, we checked that we might expect outliers from the features like Snowfall, Rainfall, Solar Radiation, and Wind Speed due to their nature
Even though we need to try handling the outliers for attempting to reach the assumptions of Linear Regression.
Handling Outliers:
The procedure of handling Outliers followed :
1.	Removing the outliers that are above the first limit ex:.99 percentile 
2.	Capping the outliers to the nearest outer fence that are below the second limit ex .98 percentile
By doing this procedure for the features like Snowfall, Rainfall, Solar Radiation, and Wind Speed the distributions that were very far (outliers) were cut off resulting in a loss of 8% to 10% of data
NOTE: Handling Outliers is only to check the performance of linear models on this type of data. In simple words to check if the linear models are suitable or not for prediction for this type of data
Even after handling a sufficient amount of outliers, only 20% of outliers were removed
Transformation:
Transforming the data and checking if we could normally distribute the data and reduce the outliers
Transformations used:
1.	Log transformation
2.	Square root transformation
3.	Exponential transformation
4.	Inverse log transformation
5.	Reciprocal transformation
6.	Exponential power transformation
After trying all these above transformations 
(1)Boxplot (2)Histogram (3)QQ plot was plotted after the transformation But the transformation dint work, the features were still not normal and we could still find a huge amount of outliers
Conclusion by analyzing the data distribution and checking the linear regression assumptions are not satisfied
The assumption of linear regression is like this:
Normal Distribution of the data was not satisfied hence from the analysis we could tell that Linear regression will not perform well and we should opt for Trees related models Like Decision Tree Random Forest that are robust to Outliers
Categorical columns :
1.	Functional Day:
a.	Functional Day as 1
b.	Non-Functional Day as 0
2.	Holiday:
a.	Holiday as 0
b.	No-Holiday as 1
3.	Seasons:
a.	Winter
b.	Autumn
c.	Summer
d.	Spring
8. Checking the performance of the linear regression :
Data treated by handling outliers is been used to expect a good performance by the Linear regression model.
8 a. Handling Colinearity: Temperature was having a high correlation with the Due point temperature, So the Temperature feature is been eliminated. 
These are the features that can be used for Linear  regression

 
Spitting the dataset(treated with outliers and checked with transformations) into train and test sets and proceeding to the next step
Training the model and calculating the results

 

Linear Regression gave a poor performance by getting an accuracy of 56%
By this we can see that Linear regression performance is optimum, Hence we should try models that are robust to outliers
9.	Checking performance with models like Random Forest, Decision Tree, AdaBoost, Gradient Boost, XGBoost
NOTE: Data set used to train these models are not undergone  a process of handling outliers
Hypertuning parameters:
1.	Decision Trees: Grid search CV was used to tune the model with cross-validation of 10 folds
Best parameter used for tuning:
a.	max depth (Best 12)
b.	max-leaf  nodes (Best: None)
c.	min samples leaf (Best:10)
d.	splitter (Best)
2. Random Forest: Random search CV technique was used for hyperparameter tuning
Best parameter used for tuning:
a.	n estimators (Best 80)
b.	max-leaf nodes (Best: None)
c.	min samples leaf (Best:9)
d.	max depth (None)
3. AdaBoost: Random search CV technique was used for hyperparameter tuning
Best parameter used for tuning:
a.	n estimators (Best 80)
b.	loss (Best square)
c.	Learning Rate (Best 0.1)
4. Gradient Boost: Random search CV technique was used for hyperparameter tuning
Best parameter used for tuning:
a.	n estimators (Best 80)
b.	min samples leaf (Best 8)
c.	max-leaf nodes (Best None)
d.	learning rate (Best 1)
e.	max features (Best 7)
5. XGBoost: Random search CV technique was used for hyperparameter tuning
Best parameter used for tuning:
a.	lambda (Best 8)
b.	max depth (Best 8)
c.	gamma (Best 2.0)
d.	learning rate (Best 1)
e.	eta  (Best 0.2)
f.	alpha (Best 1.0)
Stored all these best estimators of these above models into a dictionary 
Random Search Cross-validation is used for hyper tuning most of the models to save time 
Using these best estimators for prediction:
Results: Out of all the 5 models AdaBoost is not performing with expected accuracy more than i.e. 70% whereas all the other models are performing well 
So hence we are considering only these models :
1.	Decision Tree (80 %)
2.	Random Forest ( 85 %)
3.	Gradient Boost (84 %)
4.	XGBoost (88 %)
Metrics for Regression:
a.	MAE (mean absolute error)
b.	MSE (mean squared error)
c.	RMSE (root mean squared error)
d.	R2_score (R2_test) (score of regression model)

 
10.  Feature Importance:
These are the major features contributing most of its part for the prediction for models, They will be different for different models
Decision Tree	Random Forest	Gradient Boost	XG Boost
Temperature	Temperature	Hour	Season Winter
Hour	Hour	Season Winter	Functioning Day
Humidity	Functioning Day	Temperature	Rainfall
Solar radiation	Humidity	Rainfall	Hour
Solar Radiation	Humidity	Functioning Day	Temperature

11. Checking expectations range of each model K-Fold technique(95 % Confidence Interval)
Shuffling the dataset into n number of splits, calculating the score of cross-validated data, and storing the results. And displaying the score distribution of each model:
NOTE: Result is an array of accuracy scores obtained by each model 

1. Distribution of Accuracy score (results from cross-validation score) of Decision Tree
 .
 From the visualization of the distribution of results, we can assume  most of the scores distribution of Decision Forest for accuracy is around 80 to 90 %
2. Distribution of Accuracy score (results from cross-validation score) of Random Forest
 
From the visualization of the distribution of results, we can assume most of the distribution for the model Random Forest is around 85% to 92%
3. Distribution of Accuracy score (results from cross-validation score)  of Gradient Boost
 
From the visualization of the distribution of results, we can assume  most of the scores distribution of Gradient Boost for accuracy is around 80 to 90 %
4. Distribution of Accuracy score (results from cross-validation score) of XGBoost
 
From the visualization, we can assume  most of the scores distribution of XGBoost for accuracy is around 85 to 95 %
Confidence Interval: Here Confidence interval is calculated by taking the standard deviation of the results. 
Using these standard deviation upper range of accuracy expectation is calculated i.e. add the mean of result with 1.96 times the standard deviation of results and calculating lower range by subtracting the mean of results with 1.96 times the standard deviation of results
Upper range= mean + (1.96)*(Std.dev)
Lower range= mean - (1.96)*(Std.dev)
Why 1.96? Because The critical value for a 95% confidence interval is 1.96.
 Source

Displaying Confidence interval Charts for different models:
 

Conclusion: Linear Models Like Linear regression is not suitable to expect good performance
But Models Like XG Boost, Random Forest, Gradient Boost, Decision Tree are performing well with the confidence interval range more than 70%





