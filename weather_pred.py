#Weather forcasting using Historical Data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Temperature': [30, 32, 34, 31, 29, 28, 35, 33, 30, 31],
    'Humidity': [60, 62, 64, 58, 55, 57, 65, 63, 59, 61],
    'Wind Speed': [10, 12, 8, 11, 9, 10, 13, 12, 10, 11],
    'Precipitation': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'Next Day Temperature': [32, 34, 31, 29, 28, 35, 33, 30, 31, 32]  # Target (what we want to predict)
}

df= pd.DataFrame(data)
X= df[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation']]
y= df['Next Day Temperature']

#split data into train and test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model= LinearRegression()
model.fit(X_train, y_train)

#uses trained model to predict next days tmp
y_pred= model.predict(X_test)

mse= mean_squared_error(y_test, y_pred) #avg square dif btw actual and pred tmp's; lower mse=better performance
r2= r2_score(y_test, y_pred) #how well model fits the data; r2 score closer to 1 = better performance

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label="Actual Temperatures", marker="o")
plt.plot(y_pred, label="Predicted Temperatures", marker="x")
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature")
plt.legend() #to differentiate btw actual and pred values
plt.show()

new_data= pd.DataFrame({
    'Temperature': [30],
    'Humidity': [60],
    'Wind Speed': [10],
    'Precipitation': [0]
})

predicted_temperature= model.predict(new_data)
print(f"\n\nPredicted temperature: {predicted_temperature[0]:.2f}C")