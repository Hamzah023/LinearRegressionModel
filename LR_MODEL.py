import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

data = pd.read_csv('HousingPriceData.csv')

x = data[['area','bedrooms','bathrooms', 'stories']]
y = data[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #splits the data for testing and training, testing gets .2 and training gets .8 x test and x train and y test and y train are different

model = LinearRegression() #calling lin regr model
model.fit(x_train, y_train) #passing x and y train in model.fit to train model

y_pred = model.predict(x_test) #prediction based on test cases and model it trained and save in y pred

plt.figure(figsize=(10,6)) 
plt.scatter(y_test, y_pred, color = 'blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

plt.title('Housing Price Predictions')
plt.ylabel('Actual price')
plt.ticklabel_format(style='plain', axis='both')
plt.show()
