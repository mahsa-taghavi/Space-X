# Space-X  
 Project Overview:
 The objective of this project is to forecast the success of SpaceX rocket landings.
 The outcome variable is a binary classification defined as follows:

    Class = 0: The landing is expected to fail.
    Class = 1: The landing is anticipated to be successful.

Data Set:
The dataset utilized for this analysis is named dataset_falcon9.csv. It comprises various attributes associated 
with each rocket launch,which we will examine to gain insights into the connections between different features and the landing results.


Importing Libraries and Data:
import os
import pandas as pd
import numpy as np

# Set working directory
os.getcwd()

# Load the dataset
df = pd.read_csv('dataset_falcon9.csv')


Initial Data Exploration:

    Use the .info() method to get an overview of the dataset.

df.info()

    The dataset consists of 90 rows (0 to 89) and 18 columns, with most columns containing non-null values.
    The LandingPad column has 26 missing values.


Data Cleaning:

    Drop a specific row or column:

df.drop(2, axis=0)  # Drops the third row
df.drop('Date', axis=1)  # Drops the 'Date' column (temporary)

    Use the inplace=True parameter to make permanent changes:

df.drop('Date', axis=1, inplace=True)


Adding a Row:
To add a new row to the DataFrame, create a dictionary with the new data and use pd.concat():

new_row = {
    'FlightNumber': 11,
    'Date': '2024-01-01',
    'BoosterVersion': 'Falcon 9',
    'PayloadMass': 4500,
    'Orbit': 'LEO',
    'LaunchSite': 'CCAFS SLC 40',
    'Outcome': 'Success',
    'Flights': 10,
    'GridFins': 1,
    'Reused': 0,
    'Legs': 1,
    'LandingPad': '5e9e3032383ecb6bb234e7ca',
    'Block': 5,
    'ReusedCount': 1,
    'Serial': 'B1049',
    'Longitude': -80.577366,
    'Latitude': 28.561857,
    'Class': 1
}

df2 = pd.DataFrame(new_row, index=[0])
df2 = pd.concat([df, df2], ignore_index=True)
df2.tail()



Data Analysis:

    Analyze unique values in the BoosterVersion and PayloadMass columns:
set(df['BoosterVersion'])
min(df['PayloadMass'])
max(df['PayloadMass'])
df['PayloadMass'].mean()
df['PayloadMass'].std()

    Visualize the distribution of PayloadMass:

df['PayloadMass'].hist()


Data Visualization:
Use Matplotlib and Seaborn to visualize relationships between variables:

import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(y="PayloadMass", x="FlightNumber", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)
plt.show()


Data Preprocessing:
Convert categorical variables to numerical data and handle missing values:

df['GridFins'] = df['GridFins'].astype(int)
df['Reused'] = df['Reused'].astype(int)
df['Legs'] = df['Legs'].astype(int)

# Convert categorical columns to dummy variables
df_dummy = pd.get_dummies(df[['Orbit', 'LaunchSite', 'Outcome', 'LandingPad']])
df = pd.concat([df, df_dummy], axis=1)

# Drop original categorical columns
df.drop(['Orbit', 'LaunchSite', 'Outcome', 'LandingPad', 'Date'], axis=1, inplace=True)


Model Training with Logistic Regression:
Train a logistic regression model on the preprocessed dataset:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Create and fit the model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Make predictions
prediction = logmodel.predict(X_test)



Model Evaluation:
Evaluate the model performance using confusion matrix and accuracy score:

from sklearn.metrics import confusion_matrix, accuracy_score

confusion = confusion_matrix(y_test, prediction)
accuracy = accuracy_score(y_test, prediction, normalize=True)

print("Confusion Matrix:\n", confusion)
print("Accuracy:", accuracy)

Conclusion:
This project provides a comprehensive analysis of the dataset to predict the successful landing of SpaceX rockets using logistic regression.
The exploratory data analysis, preprocessing, and model evaluation steps are critical for understanding and improving the model's performance.

