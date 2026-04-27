
from dash import Dash, html, dcc, dash_table, Input, Output
import pandas as pd
import base64
import io
from preprocessing import Dataset

# Default Data
defaultDataset = Dataset()
df = defaultDataset.X.copy() 
df["percent_change_next_weeks_price"] = defaultDataset.y

app = Dash(__name__)
server = app.server

app.dataset = defaultDataset
app.df = df

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
        html.Label("Choose target column:"),
        dcc.Dropdown(
            id="target-dropdown",
            options=[{"label": col, "value": col} for col in df.columns],
            placeholder="Select a target variable",
            style={"width": "40%", "color": "black"}
        ),

        html.Br(),

        dcc.RadioItems(
            id="problem-type",
            options=[
                {"label": "Regression", "value": "regression"},
                {"label": "Classification", "value": "classification"},
            ],
            value="regression",
            inline=True
        ),

        html.Div(id="target-output")
    ])

@app.callback(
    Output("file-name", "children"),
    Output("data-preview", "columns"),
    Output("data-preview", "data"),
    Output("target-dropdown", "options"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename")
)
def uploadData(contents, filename):
    if contents is None:
        app.dataset = defaultDataset
    else:
        contentType, contentString = contents.split(",")
        decoded = base64.b64decode(contentString)

        uploaded_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        app.dataset = Dataset(csv_file=uploaded_df)

    app.df = app.dataset.X.copy()
    app.df["percent_change_next_weeks_price"] = app.dataset.y

    if contents is None:
        message = "Using default Dow Jones dataset."
    else:
        message = f"Uploaded file: {filename} | Rows: {app.df.shape[0]} | Columns: {app.df.shape[1]}"

    columns = [{"name": col, "id": col} for col in app.df.columns]
    table_data = app.df.head(16).to_dict("records")
    target_options = [{"label": col, "value": col} for col in app.df.columns]

    return message, columns, table_data, target_options

@app.callback(
    Output("target-output", "children"),
    Input("target-dropdown", "value"),
    Input("problem-type", "value")
)

def targetSelection(target, problem_type):
    if target is None:
        return "Select a target variable."

    if problem_type == "regression":
        average = app.df[target].mean()

        return html.Div([
            html.H4("Regression Target Summary"),
            html.P(f"Average value of {target}: {average:.4f}")
        ])

    else:
        counts = app.df[target].value_counts()

        return html.Div([
            html.H4("Classification Target Summary"),
            html.Ul([
                html.Li(f"{label}: {count}")
                for label, count in counts.items()
            ])
        ])

if __name__ == "__main__": 
    app.run(debug=True)
