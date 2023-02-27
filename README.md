# Assignment--11
**OVERVIEW**

In this application, you will explore a dataset from kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

###  problem statement

As a used car dealer, staying on top of the constantly changing market value of your cars is an essential part of running a successful business. To remain competitive in this space, predicting the true worthof each vehicle based on its attributes can be critical to accurate pricing and increased customer interest. Developing a predictive model that considers factors such as make, model year, mileage and condition could help ensure you get more bang for your buck when setting prices and attract new customers.

Several machine learning models can be used to predict the value of a user's car. The choice of model depends on several factors, including the size and quality of the available data, the complexity of the problem, and the desired level of accuracy. Some of the commonly used models for car value prediction include:

1.	Linear regression: This simple model fits a linear equation to the data to predict the car's value. It works well with a linear relationship between the input features and the output value.
2.	Random Forest: This is a powerful model that can handle both categorical and continuous features. It works well when there are many input features, and the relationship between them and the output value is complex.
3.	Gradient Boosting: This model uses an ensemble of weak learners to make predictions. It works well when there are many input features, and the relationship between them and the output value is complex.
4.	Neural Networks: This powerful model can handle complex relationships between the input features and the output value. It works well when a large amount of data is available, and the relationship between the input features and the output value is non-linear.

It is important to note that the model's accuracy depends on the quality and quantity of the data used for training. Therefore, having a large, diverse, and high-quality dataset is essential to train the model for accurate predictions.

The business problem could involve addressing the following questions:

1. How can we develop a predictive model that accurately estimates the value of a used car based on its attributes?
2. Which features are the most important in determining the value of a used car?
3. How can we validate the accuracy of the predictive model?
4. How can we use the predictive model to set competitive prices for our used cars?
5. How can we update the predictive model to reflect changes in the market and in the availability of data?

By addressing these questions, the used car dealer could develop a data-driven approach to pricing their used cars that takes into account the unique features of each car and the current market conditions. This could lead to increased customer satisfaction, increased sales, and improved profitability for the dealership.

We are exploring two innovative approaches to predicting the value of used cars from a dataset containing three million entries. The first method is linear regression, an intuitive machine learning technique which draws meaningful insights from data using mathematical models and equations. Our second approach involves KMeans(), a powerful clustering algorithm provided by scikit-learn library in Python that organizes similar data points into clusters for unsupervised learning. By randomly selecting K centroids and iteratively assigning them until convergence is reached, we can gain valuable insight about how car attributes shape their market value!

### Read the data file and analyezed 

data file vehicles.csv ahs subset of The original dataset contained information on 3 million used cars
It has number of rows - 426880 and number of columns - 18

we must remove irrelevant features like ‘id’, ’region’, ’vin’, ’title_status’, ’state’ and others from the dataset.

The results indicate many columns have null values and data types that may not provide helpful information for MLM. I analyze and apply corrections
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 
dtypes: float64(2), int64(2), object(14)
memory usage: 58.6+ MB

I applied the following data scrubbing 

UsedCar_data = UsedCar_data.convert_dtypes() this converted features Data type Object to String

Based on analyses, I drop the following features
UsedCar_data_Rev2 = UsedCar_data.drop(['id','VIN','state' ,'region','title_status'], axis=1)
UsedCar_data_Rev2 = UsedCar_data_Rev2.drop(['size'], axis=1)
UsedCar_data_Rev2 = UsedCar_data_Rev2.drop(['model'], axis=1)


UsedCar_data_r2.loc[UsedCar_data_r2['year'].isnull(),'year'] = 0
UsedCar_data_r2.loc[UsedCar_data_r2['transmission'].isnull(),'transmission'] = 'NaN_tran'
UsedCar_data_r2.loc[UsedCar_data_r2['condition'].isnull(),'condition'] = 'salvage'
UsedCar_data_r2.loc[UsedCar_data_r2['size'].isnull(),'size'] = 'Unknown_size'
UsedCar_data_r2.loc[UsedCar_data_r2['type'].isnull(),'type'] = 'NaN_type'
UsedCar_data_r2.loc[UsedCar_data_r2['manufacturer'].isnull(),'manufacturer'] = 'Unknown_man'
UsedCar_data_r2.loc[UsedCar_data_r2['model'].isnull(),'model'] = 'NaN_model'
UsedCar_data_r2.loc[UsedCar_data_r2['cylinders'].isnull(),'cylinders'] = 'NaN_cylinders'
UsedCar_data_r2.loc[UsedCar_data_r2['fuel'].isnull(),'fuel'] = 'NaN_fuel'
UsedCar_data_r2.loc[UsedCar_data_r2['odometer'].isnull(),'odometer'] = 0

aftre applying changes there are no null values

I also notice that the target features  have outlier values that may negatively impact the  MLM
I will drop features rows above and under specific values


sns.countplot(y = UsedCar_data_r2.year) showed that there are not many cars under the year 1995, so it would help to drop them
UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["year"] > 1995 ]

Keep rows with price range $3000 to $250000

UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["price"] > 3000 ]
UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["price"] < 250000 ]

## I applied a similar approach to other features

UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["type"].isin(['pickup', 'truck', 'coupe', 'SUV' ,'hatchback', 'mini-van', 'sedan', 'van' ,'convertible', 'wagon'])]
UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["cylinders"].isin(['8 cylinders','6 cylinders','4 cylinders'])]

## I execuetd One-Hot Encode for features to conevr them in usefull values

one_hot = pd.get_dummies(UsedCar_data_r2['condition'])
UsedCar_data_r2 = UsedCar_data_r2.drop('condition',axis = 1)
UsedCar_data_r2 = UsedCar_data_r2.join(one_hot)

Use same code for 'type,transmission,cylinders and others'

# Although I had put in effort to clean the dataset and applied a linear regression model, my expectations of success were not met. It appears that I need some personalized guidance to make sense of this assignment.

## Conclusion 

### Based on the analyses, I think the dataset has a problem. To create a  good MLM, we need a dataset vetted by a business analyst with expertise in the domain ( Used car pricing) to select the right features  and have a dataset that reflects reality
