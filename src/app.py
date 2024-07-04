import dash
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Finance_analyzer")

server = app.server

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="Analysis Type",
        color="rgba(0, 0, 0, 0.5)"
    ),
    brand=html.H1("Financial Analyzer"),
    color="rgba(0, 0, 0, 0.5)",
    dark=True,
    className="mb-2",
    style={'backgroundColor': 'rgba(0, 0, 0, 0.5)'}
)

footer = dbc.Container(
    html.Footer([
        html.Div([
            html.A('GitHub', href='https://github.com/SujitJustineBarwa', target='_blank', style={'color': 'white', 'marginRight': '15px'}),
            html.Span('© 2024', style={'color': 'white', 'marginRight': '15px'}),
            html.Span('Contact: sujitjustine@gmail.com', style={'color': 'white'})
        ], style={'textAlign': 'center', 'padding': '10px 0'})
    ], style={'position': 'fixed', 'left': '0', 'bottom': '0', 'width': '100%', 'backgroundColor': 'rgba(0, 0, 0, 0.5)', 'color': 'white'}),
    fluid=True,
    className="mt-5"
)

app.layout = dbc.Container(
    [navbar, dash.page_container, footer],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)