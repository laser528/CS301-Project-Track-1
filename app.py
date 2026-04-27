
from dash import Dash, html

app = Dash(__name__)
server = app.server

app.layout = html.Div(children=[
        html.H1("Machine Learning Dashboard"),

        html.P(
             "This dashboard will guide users through uploading data, "
             "choosing a target variable, training models, and making predictions on stock data."
        ),

        html.Hr(),

    ])

if __name__ == "__main__": 
    app.run(debug=True)
