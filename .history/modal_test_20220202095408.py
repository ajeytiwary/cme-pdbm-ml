import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

COOKIE = "https://todaysmama.com/.image/t_share/MTU5OTEwMzkyMDIyMTE1NzAz/cookie-monster.png"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Button("Show the cookie monster", id="button"),
        dbc.Modal(
            [
                dbc.ModalHeader("Cookies!"),
                dbc.ModalBody(html.Img(src=COOKIE, style={"width": "100%"})),
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

