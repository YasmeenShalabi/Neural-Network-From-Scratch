#Predict House Prices with Linear Regression
#Uses California housing dataset- contains info about different districts in Cali (features- population,median income, housage, more...)

import pandas as pd #handle dataset manipulation
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #to predict house prices
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Load the California housing dataset; includes features and target variable(median house value in hundreds of thousands of dollars)
housing= fetch_california_housing(as_frame=True)

#Create a DataFrame from the dataset
df= housing.frame

print("California Housing Data:")
print(df.head())

#get features(indepenent variables) and target(dependent variable)
X= df.drop('MedHouseVal', axis=1)
y= df['MedHouseVal']

#split data into training and testing
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42) #random_state to make sure split is always the same

#Train the Linear Regression Model
model= LinearRegression() #initializes linear regression model
model.fit(X_train, y_train) #trains model using training data

#Make predictions on test set
y_pred= model.predict(X_test) #uses train model to predict median house value for test set

#Evaluate the model using MSE and R2 Score
#mean square error measures avg dif btw actual and predicted house values; r2_score indicates how well the model fits the data...closer to 1 means better performance
mse= mean_squared_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

print("Model Coefficients:")
print(f"Intercept: {model.intercept_}") #displays intercepts of linear regression model
print(f"Coefficients: {model.coef_}") #displays coefficients for each feature

#Create dataframe for the coefficients with their responding feature names
coef_df= pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print("Coefficients for each feature")
print(coef_df)

#Test model with new data
new_data= pd.DataFrame({
    'MedInc': [5],
    'HouseAge': [30],
    'AveRooms': [6],
    'AveBedrms': [1],
    'Population': [500],
    'AveOccup': [3],
    'Latitude': [34.05],
    'Longitude': [-118.25]
})

predicted_price= model.predict(new_data)
print(f"\n\nPredicted House Price: ${predicted_price[0]:,.2f}")