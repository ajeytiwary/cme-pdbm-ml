import dash
import dash_core_components as dcc

import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

COOKIE = "https://todaysmama.com/.image/t_share/MTU5OTEwMzkyMDIyMTE1NzAz/cookie-monster.png"

text_markdown = "\t"
with open('README.md') as this_file:
    for a in this_file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a




app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Button("Instructions", id="button"),
        dbc.Modal(
            [
                dbc.ModalHeader("Readme!"),
                dbc.ModalBody(dcc.Markdown(text_markdown)),
            ],
            id="modal",
            is_open=False,
        ),
    ],
    className="p-5",
)


@app.callback(
    Output("modal", "is_open"),
    [Input("button", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True)

