import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, callback
from scipy.stats import skew, kurtosis
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from global_variables import directory_path
import os

dash.register_page(__name__)

# List CSV files in a folder
def list_csv_files():
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    else:
        csv_files = ["No data found"]
    return [{'label': file, 'value': file} for file in csv_files]

# Define the layout of the page using Bootstrap components
layout = dbc.Container(
    [
        html.H1("Stock Price Monitor", className="mt-4 mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("Select a stock data file:"),
                        dcc.Dropdown(id="stock-selector", options=list_csv_files(), value=None, clearable=False),
                        html.Hr(),

                        html.P("Select the time period:"),
                        dcc.Slider(
                            id='date-slider',
                            min=0,
                            max=100,
                            step=1,
                            marks={i: f'{i}%' for i in range(0, 101, 10)},
                            value=100,
                        ),
                        html.Hr(),

                        html.Div(id="stats-output", className="mt-4"),

                        dcc.Interval(id='interval-component', interval=1 * 60 * 1000, n_intervals=0)
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="stock-graph"),
                        html.Hr(),
                        dcc.Graph(id="stock-return-graph"),
                        html.Hr(),
                        dcc.Graph(id="volume-graph"),
                        html.Hr(),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="histogram-graph"),
                        html.Hr(),
                        dcc.Graph(id="histogram-graph-of-returns"),
                        html.Hr(),
                        dcc.Graph(id="volume-histogram-graph"),
                        html.Hr(),
                    ],
                    md=4,
                ),
            ],
        ),

        dbc.Row([
            html.P() for _ in range(10)
        ])
    ],
    fluid=True,
    style={"height": "100vh", "overflowY": "scroll"}
)


@callback(
    [
        Output("stock-graph", "figure"),
        Output("stock-return-graph", "figure"),
        Output("volume-graph", "figure"),
        Output("histogram-graph", "figure"),
        Output("histogram-graph-of-returns", "figure"),
        Output("volume-histogram-graph", "figure"),
        Output("stats-output", "children")
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("stock-selector", "value"),
        Input("date-slider", "value")
    ]
)
def update_graph(n_intervals, selected_csv, start_percent):
    if not selected_csv:
        return {}, {}, {}, {}, {}, {}, "No stock data file selected."

    # Load selected CSV file
    csv_path = os.path.join(directory_path, selected_csv)
    stock_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if stock_data.empty:
        return {}, {}, {}, {}, {}, {}, "Selected CSV file is empty or invalid."

    # Calculate the start date based on the slider value
    start_date = datetime.now() - timedelta(days=int((start_percent / 100) * 365 * 5))
    filtered_data = stock_data[start_date:]

    # Create the plotly figure for stock price
    stock_fig = px.line(filtered_data, x=filtered_data.index, y="Close", title=f"{selected_csv} Stock Price")

    # Create the plotly figure for stock returns
    filtered_data['Daily Return'] = filtered_data['Close'].pct_change() * 100
    stock_return_fig = px.line(filtered_data, x=filtered_data.index, y='Daily Return', title=f"{selected_csv} Daily Returns")

    # Create the plotly figure for volume data
    volume_fig = px.line(filtered_data, x=filtered_data.index, y='Volume', title=f"{selected_csv} Volume Data")

    # Create the plotly figure for histogram of stock prices
    hist_fig = px.histogram(filtered_data, x="Close", nbins=30, title=f"{selected_csv} Price Distribution")

    # Create the plotly figure for histogram of stock returns
    hist_return_fig = px.histogram(filtered_data['Daily Return'].dropna(), x='Daily Return', nbins=30, title=f"{selected_csv} Daily Returns Distribution")

    # Create the plotly figure for histogram of volume data
    hist_volume_fig = px.histogram(filtered_data, x='Volume', nbins=30, title=f"{selected_csv} Volume Distribution")

    # Calculate statistics for price data
    price_data = filtered_data["Close"].dropna()
    price_skewness = skew(price_data)
    price_kurtosis = kurtosis(price_data)
    price_dataset_size = len(price_data)

    # Calculate statistics for returns data
    return_data = filtered_data['Daily Return'].dropna()
    return_skewness = skew(return_data)
    return_kurtosis = kurtosis(return_data)
    return_dataset_size = len(return_data)

    # Calculate statistics for volume data
    volume_data = filtered_data['Volume'].dropna()
    volume_skewness = skew(volume_data)
    volume_kurtosis = kurtosis(volume_data)
    volume_dataset_size = len(volume_data)

    stats_output = html.Div(
        [
            html.H4("Statistics"),
            html.Hr(),
            html.Div(
                [
                    html.H5("Price Statistics:"),
                    html.Hr(),
                    html.P(f"Skewness: {price_skewness:.2f}"),
                    html.P(f"Kurtosis: {price_kurtosis:.2f}"),
                    html.P(f"Dataset size: {price_dataset_size}")
                ],
                style={"margin-bottom": "20px"}
            ),
            html.Div(
                [
                    html.H5("Returns Statistics:"),
                    html.Hr(),
                    html.P(f"Skewness: {return_skewness:.2f}"),
                    html.P(f"Kurtosis: {return_kurtosis:.2f}"),
                    html.P(f"Dataset size: {return_dataset_size}")
                ],
                style={"margin-bottom": "20px"}
            ),
            html.Div(
                [
                    html.H5("Volume Statistics:"),
                    html.Hr(),
                    html.P(f"Skewness: {volume_skewness:.2f}"),
                    html.P(f"Kurtosis: {volume_kurtosis:.2f}"),
                    html.P(f"Dataset size: {volume_dataset_size}")
                ]
            )
        ]
    )

    return stock_fig, stock_return_fig, volume_fig, hist_fig, hist_return_fig, hist_volume_fig, stats_output
