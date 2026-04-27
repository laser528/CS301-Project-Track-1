
from dash import Dash, html, dcc, dash_table, Input, Output
import pandas as pd
import base64
import io
from preprocessing import Dataset

# Default Data
default_data = Dataset()
default_df = default_data.X.copy()

app = Dash(__name__)
server = app.server

app.layout = html.Div(children=[
        html.H1("Machine Learning Dashboard"),
        html.P(
             "This dashboard will guide users through uploading data, "
             "choosing a target variable, training models, and making predictions on stock data."
        ),
        html.Hr(),
        dcc.Upload(
            id="upload-data",
            children=html.Div([ html.A("Upload CSV File")]),
            style={"width": "40%","height": "60px","lineHeight": "60px","borderWidth": "2px","borderStyle": "dashed","borderRadius": "10px","textAlign": "center","marginBottom": "20px",},
            multiple=False
        ),
        html.Div(id="file-name"),
        html.H3("Dataset Preview"),
        dash_table.DataTable(
            id="data-preview",
            page_size=5,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
        ),

        
    ])

@app.callback(
    Output("file-name", "children"),
    Output("data-preview", "columns"),
    Output("data-preview", "data"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename")
)
def load_data(contents, filename):
    if contents is None:
        # DOW JONES PULLED FROM THE INTERNET i think ??? 
        df = default_df
        columns = [{"name": col, "id": col} for col in df.columns]
        data = df.head(16).to_dict("records")
        return ("Using default Dow Jones preprocessed dataset.",columns,data)

    # DATA PULLED FROM CSV im unsure how you want this to relate to the preprocessing class as it dosent take any parameters
    contentType, contentString = contents.split(",")
    decoded = base64.b64decode(contentString)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    columns = [{"name": col, "id": col} for col in df.columns]
    data = df.head(16).to_dict("records")

    return (f"Uploaded file: {filename} | Rows: {df.shape[0]} | Columns: {df.shape[1]}", columns, data)


if __name__ == "__main__": 
    app.run(debug=True)
