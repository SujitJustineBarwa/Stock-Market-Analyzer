import dash
from dash import html
from dash import dcc, Input, Output, callback
import os
import yfinance as yf
import pandas as pd
from global_variables import directory_path,stock_symbols  # Assuming directory_path is defined elsewhere

# Function to fetch and save stock data to CSV
def fetch_and_save_stock_data(symbol, frequency='1d'):
    # Fetch data using yfinance
    stock_data = yf.download(symbol, period='max', interval=frequency)
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    # Save data to CSV
    csv_filename = f"{directory_path}/{symbol}_stock_data_{frequency}.csv"
    stock_data.to_csv(csv_filename)
    
    return csv_filename

# Register the Dash app page
dash.register_page(__name__, path="/")

# Layout for the Dash app (modified)
layout = html.Div(
    [
        html.Button("Fetch Stock Data", id="fetch-button"),
        html.Button("Clear Stock Data", id="clear-button"),  # New button
        html.Hr(),

       html.P("Select the frequency:"),
       dcc.Dropdown(
           id='frequency-selector',
           options=[
               {'label': 'Daily', 'value': '1d'},
               {'label': 'Weekly', 'value': '1wk'},
               {'label': 'Monthly', 'value': '1mo'},
               {'label': 'Quarterly', 'value': '3mo'}
           ],
           value='1d',
           style={'width': '100%'},
           clearable=False
       ),
       html.Hr(),

        html.Div(id="output-data")
    ]
)

# Callback to fetch and display stock data (modified)
@callback(
    Output("output-data", "children"),
    [Input("fetch-button", "n_clicks"),
     Input("clear-button", "n_clicks"),  # New input
     Input("frequency-selector", "value")]
)
def fetch_stock_data(fetch_clicks, clear_clicks, frequency):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # Check which button triggered the callback

    if 'fetch-button' in changed_id:    
        if fetch_clicks is None:
            return ""
        filenames = []
        for symbol in stock_symbols:
            csv_filename = fetch_and_save_stock_data(symbol, frequency)
            filenames.append(html.P(f"Saved {symbol} data ({frequency}) to {csv_filename}"))
        return filenames
        
    elif 'clear-button' in changed_id:
        if clear_clicks is not None:  # Only clear if the button was clicked
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            return html.P("Stock data cleared!")    