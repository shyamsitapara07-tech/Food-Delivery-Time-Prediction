# Food Delivery Time Prediction Project

# Import the Python Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Data Collection

df = pd.read_csv('Dataset/Food_Delivery_Time_Dataset.csv')
print(df)



# Data Cleaning & Data cleaning

print("missing values: ",df.isnull().sum())
print("duplicate values: ",df.duplicated().sum())
print("unique values: ",df.nunique())

print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
qty = df.select_dtypes(include=['int64','float']).columns
print(qty)

qlty = df.select_dtypes(include=['object','category']).columns
print(qlty)



print("missing values: ",df.isnull().sum())
print("duplicate values: ",df.duplicated().sum())
print("unique values: ",df.nunique())


# Handling Missing values

df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Traffic_Level'] = df['Traffic_Level'].fillna(df['Traffic_Level'].mode()[0])
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(df['Courier_Experience_yrs'].median())
print("missing values: ",df.isnull().sum())




# Handling Outliers

plt.figure(figsize=(8, 5))
plt.boxplot(df['Delivery_Time_min'])
plt.ylabel('Delivery Time (minutes)')
plt.title('Box Plot of Delivery Time')
plt.savefig('Box_Plot_Before.png')
plt.show()


# Handle the outliers of Delivery_Time_min

Q1=df['Delivery_Time_min'].quantile(0.25)
Q3=df['Delivery_Time_min'].quantile(0.75)

IQR = Q3 - Q1

# set the limit

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Delivery_Time_min'] >= lower_bound) & (df['Delivery_Time_min'] <= upper_bound)]


# Exploratory Data Analysis 



# Histogram

plt.figure(figsize=(8, 5))
plt.hist(df['Delivery_Time_min'], bins=20)
plt.xlabel('Delivery Time (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Delivery Time')
plt.savefig('Histogram.png')
plt.show()


# Boxplot

plt.figure(figsize=(8, 5))
plt.boxplot(df['Delivery_Time_min'])
plt.ylabel('Delivery Time (minutes)')
plt.title('Box Plot of Delivery Time')
plt.savefig('Box_Plot_After.png')
plt.show()


# Pie chart

weather_counts = df['Weather'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    weather_counts,
    labels=weather_counts.index,
    autopct='%1.1f%%'
)
plt.title('Weather Distribution')
plt.savefig('Pie_Chart.png')
plt.show()


# Bar chart

traffic_delivery = df.groupby('Traffic_Level')['Delivery_Time_min'].mean()

plt.figure(figsize=(8, 5))
plt.bar(traffic_delivery.index, traffic_delivery.values)
plt.xlabel('Traffic Level')
plt.ylabel('Average Delivery Time (minutes)')
plt.title('Average Delivery Time by Traffic Level')
plt.savefig('Bar_Chart.png')
plt.show()


# Coorelation Matrix

import seaborn as sns

numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('Heatmap.png')
plt.show()




# Automated EDA

import sweetviz as sv

report =  sv.analyze(df)
report.show_html()


# one hot coding (Convert string column to numeric)

df = pd.get_dummies(df,columns=['Weather'],dtype=int)
df = pd.get_dummies(df,columns=['Traffic_Level'],dtype=int)
df = pd.get_dummies(df,columns=['Time_of_Day'],dtype=int)
df = pd.get_dummies(df,columns=['Vehicle_Type'],dtype=int)



# Feature SelecƟon 


# Dependent & Independent variables

X = df[['Distance_km','Preparation_Time_min','Courier_Experience_yrs','Weather_Clear','Weather_Foggy','Weather_Rainy','Weather_Snowy','Weather_Windy','Traffic_Level_High','Traffic_Level_Low','Traffic_Level_Medium','Time_of_Day_Afternoon','Time_of_Day_Evening','Time_of_Day_Morning','Time_of_Day_Night','Vehicle_Type_Bike','Vehicle_Type_Car','Vehicle_Type_Scooter']]

y = df['Delivery_Time_min']

print(X.shape)
print(y.shape)


# Train-Test Split 


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# Build ML Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

print('Model build successfully!')




# Predct the X_test

y_pred = model.predict(X_test)
print(y_pred)



# Model Evaluation

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5
r2 = r2_score(y_test,y_pred)

print("MAE: ",mae)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("R2: ",r2)



# Improve Model



# Cross Validation

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("Cross Validation R² Scores:")
print(cv_scores)

print("Mean R²:", cv_scores.mean())



# Ridge Model

from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# analyze ridge


from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score
)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Ridge Regression")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)