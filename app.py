
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

        # dcc.RadioItems(
        #     id="problem-type",
        #     options=[
        #         {"label": "Regression", "value": "regression"},
        #         {"label": "Classification", "value": "classification"},
        #     ],
        #     value="regression",
        #     inline=True
        # ),

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
        app.raw_df = df
    else:
        contentType, contentString = contents.split(",")
        decoded = base64.b64decode(contentString)

        uploadedDF = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        app.raw_df = uploadedDF
        app.dataset = Dataset(csvFile=uploadedDF)

    app.df = app.dataset.X.copy()
    app.df[app.dataset.TargetColumn] = app.dataset.y

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
    Input("target-dropdown", "value")
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
        if feature == target:
            continue

        if pd.api.types.is_numeric_dtype(app.df[feature]):
            corr = app.df[feature].corr(app.df[target])
            correlations.append({
                "Feature": feature,
                "Correlation": corr,
                "Type": "Valid"
            })
        else:
            correlations.append({
                "Feature": feature,
                "Correlation": 0,
                "Type": "Ignored"
            })

    corrDF = pd.DataFrame(correlations)

    fig = px.bar(
        corrDF,
        x="Feature",
        y="Correlation",
        color="Type",
        title=f"Correlation with {target}",
        color_discrete_map={
            "Valid": "green",
            "Ignored": "gray"
        }
    )

    return fig

@app.callback(
    Output("model-results-output", "children"),
    Input("target-dropdown", "value")
)
def showModelResults(target):
    if target is None:
        return "Select a target variable to train the model."

    try:
        app.dataset = Dataset(csvFile=app.raw_df, targetColumn=target)

        return html.Div([
            html.H4("Model Results"),
            html.P(f"Target: {target}"),
            html.P(f"R² Score: {app.dataset.r2Score:.4f}"),
            html.P(f"RMSE: {app.dataset.RMSE:.4f}")
        ])

    except Exception as e:
        return html.Div([
            html.H4("Model Training Error"),
            html.P(str(e))
        ])

@app.callback(
    Output("prediction-output", "children"),
    Input({"type": "prediction-input", "index": ALL}, "value"),
    State({"type": "prediction-input", "index": ALL}, "id")
)
def makePrediction(values, ids):
    if not values or not ids:
        return ""

    # Start with average values from the trained dataset
    input_row = app.dataset.X.mean(numeric_only=True).to_dict()

    # Override averages with user-entered values
    for value, input_id in zip(values, ids):
        feature_name = input_id["index"]

        if value is not None:
            input_row[feature_name] = value

    predictionDF = pd.DataFrame([input_row])
    predictionDF = predictionDF.reindex(columns=app.dataset.X.columns, fill_value=0)

    try:
        prediction = app.dataset.model.predict(predictionDF)[0]

        if hasattr(prediction, "__len__"):
            prediction = prediction[0]

        return html.Div([
            html.H4("Prediction Result"),
            html.P(f"Predicted {app.dataset.TargetColumn}: {prediction:.4f}")
        ])

    except Exception as e:
        return html.Div([
            html.H4("Prediction Error"),
            html.P(str(e))
        ])

@app.callback(
    Output("prediction-inputs", "children"),
    Input("target-dropdown", "value")
)
def createPredictionInputs(target):
    if target is None:
        return "Select a target variable first."

    inputs = []

    for feature in app.raw_df.columns:
        if feature == target:
            continue

        if not pd.api.types.is_numeric_dtype(app.raw_df[feature]):
            continue

        inputs.append(html.Div([
            html.Label(feature),
            dcc.Input(
                id={"type": "prediction-input", "index": feature},
                type="number",
                value=app.raw_df[feature].mean(),
                style={"marginBottom": "10px", "display": "block"}
            )
        ]))

    return inputs
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)