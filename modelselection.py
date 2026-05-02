from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import plotly.express as px
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class BestModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def bestModel(self,linearR2, linearRMSE):
        decision, decisionR2, decisionRMSE = self.decisionModel()
        polynomial, polynomialR2, polynomialRMSE = self.polynomialModel()
        
        if decisionRMSE < linearRMSE and decisionRMSE < polynomialRMSE:
            return decision, decisionR2, decisionRMSE
        elif polynomialRMSE < linearRMSE and polynomialRMSE < decisionRMSE:
            return polynomial, polynomialR2, polynomialRMSE
        
        return None #If neither are better, return None signifies to stick with Linear Regression

        
    def decisionModel(self):
        regr = DecisionTreeRegressor()
        param_grid = {
            'max_leaf_nodes': [2,3,4,5,6,8,9,10]
        }

        grid_search = GridSearchCV(
            estimator=regr,
            param_grid=param_grid,
            cv = 5
        )

        grid_search.fit(self.X_train, self.y_train)

        y_pred = grid_search.predict(self.X_test)

        r2Score = r2_score(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)

        return grid_search, r2Score, RMSE
    
    def polynomialModel(self):
        pipe = Pipeline(steps=[
            ('poly', PolynomialFeatures()),
            ('model', LinearRegression())
        ])

        param_grid = {
            'poly__degree': [2, 3],
            'poly__include_bias': [True, False]
        }

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=5,
            scoring='r2'
        )

        grid_search.fit(self.X_train, self.y_train)

        y_pred = grid_search.predict(self.X_test)

        r2Score = r2_score(self.y_test, y_pred)
        RMSE = root_mean_squared_error(self.y_test, y_pred)

        return grid_search, r2Score, RMSE

if __name__ == "__main__":
    from preprocessing import Dataset

    data = Dataset()
    models = BestModel(
        data.X_train,
        data.X_test,
        data.y_train,
        data.y_test
    )

    _, decisionr2, decisionRMSE = models.decisionModel()
    _, polyr2, polyRMSE = models.polynomialModel()

    evaluation = pd.DataFrame({
        "Model" : ["Linear Regression", "Decision Tree", "Polynomial Regression"],
        "R2 Score" : [data.r2Score, decisionr2, polyr2],
        'RMSE' : [data.RMSE, decisionRMSE, polyRMSE]
    })

    fig_r2 = px.bar(evaluation, x="Model", y="R2 Score", title="Model R2 Scores")
    fig_rmse = px.bar(evaluation, x="Model", y="RMSE", title="Model RMSE Scores")

    fig_r2.show()
    fig_rmse.show()