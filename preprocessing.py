#!pip install ucimlrepo                                         <-- Use in Google Colab
#python -m pip install ucimlrepo                                <-- Use in terminal
#python -m pip install ucimlrepo scikit-learn pandas numpy      <-- Use in terminal

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# fetch dataset
dow_jones_index = fetch_ucirepo(id=312)

# data (as pandas dataframes)
X = dow_jones_index.data.features.copy()
y = dow_jones_index.data.targets.copy()

#Format dollar amounts to floats
X['open'] = X['open'].str.replace('$', '', regex=False).astype(float)
X['high'] = X['high'].str.replace('$', '', regex=False).astype(float)
X['low'] = X['low'].str.replace('$', '', regex=False).astype(float)
X['close'] = X['close'].str.replace('$', '', regex=False).astype(float)
X['next_weeks_open'] = X['next_weeks_open'].str.replace('$', '', regex=False).astype(float)
X['next_weeks_close'] = X['next_weeks_close'].str.replace('$', '', regex=False).astype(float)

#Fill in missing values
X = X.fillna(X.mode())

floatX = X[['open','high','low','close','next_weeks_open','next_weeks_close']] #Filter out non-numeric categories

X_train, X_test, y_train, y_test = train_test_split(floatX,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

from sklearn.metrics import r2_score
r2Score = r2_score(y_test, y_predicted)

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(y_test, y_predicted)

print("R2 Score: ", r2Score)
print("Root Mean Squared Error Score: ", RMSE)
