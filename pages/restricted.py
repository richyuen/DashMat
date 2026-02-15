"""Access restricted page for role-gated routes."""

from urllib.parse import parse_qs, unquote_plus

import dash_mantine_components as dmc
from dash import Input, Output, callback, clientside_callback, dcc, register_page

register_page(__name__, path="/restricted", name="Restricted", title="Access Restricted")


def _decode_target(search: str | None) -> str:
    if not search:
        return "this page"
    query = search[1:] if search.startswith("?") else search
    target = parse_qs(query).get("target", [""])[0]
    decoded_target = unquote_plus(target).strip()
    return decoded_target or "this page"


layout = dmc.Container(
    size="sm",
    style={
        "minHeight": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "center",
    },
    children=[
        dcc.Location(id="dashmat-restricted-url", refresh=False),
        dcc.Location(id="dashmat-restricted-nav", refresh=False),
        dmc.Stack(
            gap="xs",
            w="100%",
            children=[
                dmc.Text(
                    "Access Restricted",
                    variant="gradient",
                    gradient={"from": "red", "to": "orange", "deg": 90},
                    style={"fontSize": "40px", "lineHeight": 1.1},
                ),
                dmc.Divider(),
                dmc.Text(id="dashmat-restricted-message", size="lg"),
                dmc.Text(
                    "Please contact the administrator to request access.",
                    size="md",
                    c="dimmed",
                ),
                dmc.Space(h="sm"),
                dmc.Button(
                    "Return to Home",
                    id="dashmat-restricted-return-home-btn",
                    color="orange",
                ),
            ],
        ),
    ],
)


@callback(
    Output("dashmat-restricted-message", "children"),
    Input("dashmat-restricted-url", "search"),
)
def update_restricted_message(search):
    target = _decode_target(search)
    return f"Your account does not have access to {target}."


clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            window.location.href = '/';
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dashmat-restricted-nav", "pathname"),
    Input("dashmat-restricted-return-home-btn", "n_clicks"),
    prevent_initial_call=True,
)
