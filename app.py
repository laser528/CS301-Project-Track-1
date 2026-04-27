
from dash import Dash, html, dcc, dash_table, Input, Output, State, ALL
import pandas as pd
import base64
import io
from preprocessing import Dataset
import plotly.express as px

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

        html.Div(id="target-output"),

        html.Hr(),

        html.H2("Feature Selection & Visualization"),

        html.Label("Select feature columns:"),

        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": col, "value": col} for col in df.columns],
            multi=True,
            placeholder="Select feature columns",
            style={"width": "60%", "color": "black"}
        ),

        dcc.Graph(id="correlation-graph"),

        html.Hr(),

        html.H2("Model Training Results"),

        html.Div(id="model-results-output"),

        html.Hr(),

        html.H2("Prediction Interface"),

        html.P("Enter values for the selected features to make a prediction."),

        html.Div(id="prediction-inputs"),

        html.Div(id="prediction-output")
    ])

@app.callback(
    Output("file-name", "children"),
    Output("data-preview", "columns"),
    Output("data-preview", "data"),
    Output("target-dropdown", "options"),
    Output("feature-dropdown", "options"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename")
)
def uploadData(contents, filename):
    if contents is None:
        app.dataset = defaultDataset
    else:
        contentType, contentString = contents.split(",")
        decoded = base64.b64decode(contentString)

        uploadedDF = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        app.dataset = Dataset(csv_file=uploadedDF)

    app.df = app.dataset.X.copy()
    app.df["percent_change_next_weeks_price"] = app.dataset.y

    if contents is None:
        message = "Using default Dow Jones dataset."
    else:
        message = f"Uploaded file: {filename} | Rows: {app.df.shape[0]} | Columns: {app.df.shape[1]}"

    columns = [{"name": col, "id": col} for col in app.df.columns]
    tableData = app.df.head(16).to_dict("records")
    targetOptions = [{"label": col, "value": col} for col in app.df.columns]
    featureOptions = [{"label": col, "value": col} for col in app.df.columns]

    return message, columns, tableData, targetOptions, featureOptions

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
    

@app.callback(
    Output("correlation-graph", "figure"),
    Input("feature-dropdown", "value"),
    Input("target-dropdown", "value")
)
def featureCorrelation(selected_features, target):
    if not selected_features or target is None:
        return px.bar(title="Select features and a target variable.")

    correlations = []

    for feature in selected_features:
        if feature != target and pd.api.types.is_numeric_dtype(app.df[feature]):
            corr = app.df[feature].corr(app.df[target])
            correlations.append({"Feature": feature, "Correlation": corr})

    if not correlations:
        return px.bar(title="No numeric features selected for correlation.")

    corrDF = pd.DataFrame(correlations).sort_values(by="Correlation", ascending=False)

    fig = px.bar(corrDF, x="Feature", y="Correlation", title=f"Correlation with {target}")

    return fig

@app.callback(
    Output("model-results-output", "children"),
    Input("upload-data", "contents")
)
def showModelResults(contents):

    return html.Div([
        html.H4("Linear Regression Model Results"),
        html.P(f"R² Score: {app.dataset.r2Score:.4f}"),
        html.P(f"RMSE: {app.dataset.RMSE:.4f}")
    ])

@app.callback(
    Output("prediction-inputs", "children"),
    Input("feature-dropdown", "value")
)
def createPredictionInputs(selected_features):
    if not selected_features:
        return "Select features first."

    inputs = []

    for feature in selected_features:
        if pd.api.types.is_numeric_dtype(app.df[feature]):
            inputs.append(html.Div([
                html.Label(feature),
                dcc.Input(
                    id={"type": "prediction-input", "index": feature},
                    type="number",
                    placeholder=f"Enter {feature}",
                    style={"marginBottom": "10px", "display": "block"}
                )
            ]))

    return inputs

@app.callback(
    Output("prediction-output", "children"),
    Input({"type": "prediction-input", "index": ALL}, "value"),
    State({"type": "prediction-input", "index": ALL}, "id")
)
def makePrediction(values, ids):
    if not values or not ids:
        return ""

    input_row = app.dataset.X.mean(numeric_only=True).to_dict()

    for value, input_id in zip(values, ids):
        feature_name = input_id["index"]

        if value is None:
            value = 0

        input_row[feature_name] = value

    predictionDF = pd.DataFrame([input_row])
    predictionDF = predictionDF.reindex(columns=app.dataset.X.columns, fill_value=0)

    prediction = app.dataset.model.predict(predictionDF)[0]

    if hasattr(prediction, "__len__"):
        prediction = prediction[0]

    return html.Div([
        html.H4("Prediction Result"),
        html.P(f"Predicted % change next week: {prediction:.2f}%")
    ])

if __name__ == "__main__": 
    app.run(debug=True)
