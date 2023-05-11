import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns
from dash.dependencies import Input, Output
from flask_caching import Cache
import uuid
import os
import time

app = dash.Dash()
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "redis",
        # Note that filesystem cache doesn't work on systems with ephemeral
        # filesystems like Heroku.
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "cache-directory",
        # should be equal to maximum number of users on the app at a single time
        # higher numbers will store more data in the filesystem / redis cache
        "CACHE_THRESHOLD": 200,
    },
)
curr_time_stamp = 0

data_cache = {}


def serve_layout():
    session_id = str(uuid.uuid4())
    data_cache[session_id] = {"timestamp": 0}
    return html.Div(
        children=[
            dcc.Store(data=session_id, id="session-id"),
            html.H1(children="Forest Fire Tracker"),
            html.Img(
                id="image",
                src=app.get_asset_url(f"{curr_time_stamp}.png"),
                style={"height": "70vh"},
            ),
            dcc.Interval(
                id="interval-component",
                interval=1.5 * 1000,  # in milliseconds
                n_intervals=0,
            )
            # dcc.Graph(id="fare_vs_age", figure=fig)
        ]
    )


app.layout = serve_layout


@app.callback(
    Output("image", "src"),
    Input("interval-component", "n_intervals"),
    Input("session-id", "data"),
)
def update_timestamp(n, session_id):
    next = data_cache[session_id]["timestamp"] + 1
    if os.path.exists(f"assets/{next}.png") and (
        time.time() - os.path.getmtime(f"assets/{next}.png") > 2
    ):
        data_cache[session_id]["timestamp"] += 1
    return app.get_asset_url(f"{next}.png")


if __name__ == "__main__":
    app.run_server(debug=False)
