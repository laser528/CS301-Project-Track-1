
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
app.raw_df = defaultDataset.raw_df


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

 

        html.Div([
            html.Div([
                html.H3("Correlation Strength"),
                dcc.Graph(
                    id="correlation-graph",
                    style={"height": "450px"}
                ),
            ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),

            html.Div([
                html.H3("Average Target by Category"),
                dcc.Graph(
                    id="category-bar-chart",
                    style={"height": "450px"}
                ),
                dcc.Dropdown(
                    id="category-radio",
                    placeholder="Select categorical variable",
                    style={"width": "80%", "color": "black", "margin": "0 auto"}
                ),
            ], style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
        ]),

        html.Hr(),

        html.H2("Model Training Results"),

        html.Label("Select feature columns:"),

        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": col, "value": col} for col in df.columns],
            multi=True,
            placeholder="Select feature columns",
            style={"width": "60%", "color": "black"}
        ),

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
    Output("category-radio", "options"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename")
)
def uploadData(contents, filename):
    if contents is None:
        app.dataset = defaultDataset
        app.raw_df = defaultDataset.raw_df
    else:
        contentType, contentString = contents.split(",")
        decoded = base64.b64decode(contentString)

        uploadedDF = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        app.raw_df = uploadedDF
        numeric_cols_uploaded = uploadedDF.select_dtypes(include="number").columns

        if len(numeric_cols_uploaded) == 0:
            raise ValueError("Uploaded CSV needs at least one numeric column for regression.")

        default_target = numeric_cols_uploaded[-1]
        app.dataset = Dataset(csvFile=uploadedDF, targetColumn=default_target)

    app.df = app.dataset.X.copy()
    app.df[app.dataset.TargetColumn] = app.dataset.y

    if contents is None:
        message = "Using default Dow Jones dataset."
    else:
        message = f"Uploaded file: {filename} | Rows: {app.df.shape[0]} | Columns: {app.df.shape[1]}"

    columns = [{"name": col, "id": col} for col in app.df.columns]
    tableData = app.df.head(16).to_dict("records")
    
    numeric_cols = app.df.select_dtypes(include="number").columns
    categorical_cols = app.raw_df.select_dtypes(exclude="number").columns

    targetOptions = [{"label": col, "value": col} for col in numeric_cols]
    featureOptions = [{"label": col, "value": col} for col in numeric_cols if col != app.dataset.TargetColumn]
    categoryOptions = [{"label": col, "value": col} for col in categorical_cols]

    return message, columns, tableData, targetOptions, featureOptions, categoryOptions

@app.callback(
    Output("category-bar-chart", "figure"),
    Input("category-radio", "value"),
    Input("target-dropdown", "value")
)
def categoryBarChart(category, target):
    if category is None or target is None:
        return px.bar(title="Select a categorical variable and target.")

    if category not in app.raw_df.columns:
        return px.bar(title="Invalid category selected.")

    tempDF = app.raw_df.copy()

    if target not in tempDF.columns:
        tempDF[target] = app.df[target]

    grouped = tempDF.groupby(category)[target].mean().reset_index()

    fig = px.bar(
        grouped,
        x=category,
        y=target,
        title=f"Average {target} by {category}"
    )

    return fig

@app.callback(
    Output("target-output", "children"),
    Input("target-dropdown", "value")
)
def targetSelection(target):
    if target is None:
        return "Select a target variable."

    average = app.df[target].mean()

    return html.Div([
        html.H4("Regression Target Summary"),
        html.P(f"Average value of {target}: {average:.4f}")
    ])
    

@app.callback(
    Output("correlation-graph", "figure"),
    Input("target-dropdown", "value")
)
def featureCorrelation(target):
    if target is None:
        return px.bar(title="Select a target variable.")

    numeric_cols = app.df.select_dtypes(include="number").columns

    correlations = []

    for feature in numeric_cols:
        if feature == target:
            continue

        corr = app.df[feature].corr(app.df[target])

        correlations.append({
            "Feature": feature,
            "Correlation": abs(corr)
        })

    corrDF = pd.DataFrame(correlations).sort_values(
        by="Correlation",
        ascending=False
    )

    fig = px.bar(
        corrDF,
        x="Feature",
        y="Correlation",
        title=f"Correlation Strength of Numerical Variables with {target}"
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
    Input("target-dropdown", "value"),
    Input("feature-dropdown", "value")
)
def createPredictionInputs(target, selected_features):
    if target is None:
        return "Select a target variable first."

    if not selected_features:
        return "Select feature columns first."

    inputs = []

    for feature in selected_features:
        inputs.append(html.Div([
            html.Label(feature),
            dcc.Input(
                id={"type": "prediction-input", "index": feature},
                type="number",
                value=app.df[feature].mean(),
                style={"marginBottom": "10px", "display": "block"}
            )
        ]))

    return inputs
@app.callback(
    Output("feature-dropdown", "options", allow_duplicate=True),
    Input("target-dropdown", "value"),
    prevent_initial_call=True
)
def updateFeatureOptions(target):
    if target is None:
        return []

    numeric_cols = app.df.select_dtypes(include="number").columns

    return [
        {"label": col, "value": col}
        for col in numeric_cols
        if col != target
    ]


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
    # app.run(debug=True)