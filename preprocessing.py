#!pip install ucimlrepo                                         <-- Use in Google Colab
#python -m pip install ucimlrepo                                <-- Use in terminal
#python -m pip install ucimlrepo scikit-learn pandas numpy      <-- Use in terminal

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from modelselection import BestModel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

class Dataset:
    def __init__(self, csvFile = None):
        # fetch dataset
        if csvFile == None:
            dow_jones_index = fetch_ucirepo(id=312)
            # data (as pandas dataframes)
            X = dow_jones_index.data.features.copy()
            y = dow_jones_index.data.targets.copy()
        else:
            df = csvFile.copy()
            y = df["percent_change_next_weeks_price"]
            X = df.drop(columns=["percent_change_next_weeks_price"])
        

        #Format dollar amounts to floats
        X['open'] = X['open'].str.replace('$', '', regex=False).astype(float)
        X['high'] = X['high'].str.replace('$', '', regex=False).astype(float)
        X['low'] = X['low'].str.replace('$', '', regex=False).astype(float)
        X['close'] = X['close'].str.replace('$', '', regex=False).astype(float)
        X['next_weeks_open'] = X['next_weeks_open'].str.replace('$', '', regex=False).astype(float)
        X['next_weeks_close'] = X['next_weeks_close'].str.replace('$', '', regex=False).astype(float)
        X = X.fillna(X.mode().iloc[0]) #Fill in missing values

        X = pd.get_dummies(X,columns=X.select_dtypes(exclude='number').columns,drop_first=True) #One-hot encoding for non-numeric

        #Feature selection
        selector = SelectKBest(f_regression, k=int(X.shape[1]/2))
        selector.fit_transform(X, y)
        selected_columns = X.columns[selector.get_support(indices=True)]
        X = X[selected_columns]
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_predicted = model.predict(X_test)

        #Evaluate
        r2Score = r2_score(y_test, y_predicted)
        RMSE = root_mean_squared_error(y_test, y_predicted)
        
        #Pick the best model
        best = BestModel(X_train, X_test, y_train, y_test)
        if not best.bestModel(r2Score, RMSE) == None:
            model, r2Score, RMSE = best.bestModel(r2Score, RMSE)

        #Display relationship between predictor and target variables
        together = X.copy()
        together['percent_change_next_weeks_price'] = y
        numberfig = px.scatter(together, x = X.select_dtypes(include='number').columns,y='percent_change_next_weeks_price')
        nonfig = px.scatter(together, x = X.select_dtypes(exclude='number').columns,y='percent_change_next_weeks_price')

        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.r2Score = r2Score
        self.RMSE = RMSE
        self.numberfig = numberfig
        self.nonfig = nonfig

if __name__ == "__main__":
    data = Dataset()
    print("R2 Score: ", data.r2Score)
    print("Root Mean Squared Error: ", data.RMSE)
    data.numberfig.show()
    data.nonfig.show()
