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




we must remove irrelevant features like ‘id’, ’region’, ’vin’, ’title_status’, ’state’ and others from the dataset.

|    | features     | dtypes   |   NaN count |   NaN percentage |
|---:|:-------------|:---------|------------:|-----------------:|
|  0 | price        | int64    |           0 |       0          |
|  1 | year         | float64  |        1205 |       0.00282281 |
|  2 | manufacturer | object   |       17646 |       0.0413371  |
|  3 | model        | object   |        5277 |       0.0123618  |
|  4 | condition    | object   |      174104 |       0.407852   |
|  5 | cylinders    | object   |      177678 |       0.416225   |
|  6 | fuel         | object   |        3013 |       0.00705819 |
|  7 | odometer     | float64  |        4400 |       0.0103073  |
|  8 | transmission | object   |        2556 |       0.00598763 |
|  9 | size         | object   |      306361 |       0.717675   |
| 10 | type         | object   |       92858 |       0.217527   |

The results indicate many columns have null values. I analyze and apply corrections

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

sns.countplot(y = UsedCar_data_r2.year) showed that there are not many cars under the year 1995, so it would help to drop them

UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["year"] > 1995 ]

## I applied a similar approach to other features

UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["type"].isin(['pickup', 'truck', 'coupe', 'SUV' ,'hatchback', 'mini-van', 'sedan', 'van' ,'convertible', 'wagon'])]
UsedCar_data_r2 = UsedCar_data_r2.loc[UsedCar_data_r2["cylinders"].isin(['8 cylinders','6 cylinders','4 cylinders'])]

## I execuetd One-Hot Encode for features to conevr them in usefull values

one_hot = pd.get_dummies(UsedCar_data_r2['condition'])
UsedCar_data_r2 = UsedCar_data_r2.drop('condition',axis = 1)
UsedCar_data_r2 = UsedCar_data_r2.join(one_hot)

Use same code for 'type,transmission,cylinders and others'

# Although I had put in effort to clean the dataset and applied a linear regression model, my expectations of success were not met. It appears that I need some personalized guidance to make sense of this assignment.
