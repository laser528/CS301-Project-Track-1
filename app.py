
from dash import Dash, html

app = Dash(__name__)
server = app.server

app.layout = html.Div("Hello Dash")

if __name__ == "__main__": 
    app.run(debug=True)
