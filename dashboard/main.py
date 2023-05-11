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
import json
import dash_bootstrap_components as dbc
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 200
})
curr_time_stamp = 0

# data_cache = {}


def get_updated_layout(timestamp):
    stats = {}
    with open(f"assets/{timestamp}.json") as stats_f:
        stats = json.load(stats_f) 
    return [
        html.H1(children='Forest Fire Tracker'),
        dbc.Row([
            dbc.Col(html.Img(id="image", src=app.get_asset_url(f"{timestamp}.png")),width="6"),
            dbc.Col(dbc.Card(
                [
                    # dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
                    dbc.CardBody(
                        [
                            html.H1(stats["alive"], className="card-title"),
                            html.P(
                                "Alive",
                                className="card-text",
                            ),
                            
                        ]
                    ),
                ],
                style={"width": "18rem"},
            ))
        ],style={"maxHeight":"90%"})
        
    ]

def serve_layout():
    session_id = str(uuid.uuid4())
    cache.set(session_id,0)
    return html.Div(children=[
            dcc.Store(data=session_id, id='session-id'),
            html.Div(id="content",children=get_updated_layout(cache.get(session_id))),
            dcc.Interval(
                id='interval-component',
                interval=1.5*1000,  # in milliseconds
                n_intervals=0,
            )
            # dcc.Graph(id="fare_vs_age", figure=fig)
        ],style={"maxHeight":"100%"}
    )


app.layout = serve_layout

@app.callback(Output('content', 'children'),
              Input('interval-component', 'n_intervals'),
              Input('session-id', 'data'))
def update_timestamp(n,session_id):
    next_ts = cache.get(session_id) + 1
    if(os.path.exists(f"assets/{next_ts}.png") and (time.time()-os.path.getmtime(f"assets/{next_ts}.png")>2)):
        cache.set(session_id,next_ts)
    return get_updated_layout(cache.get(session_id))
    


if __name__ == "__main__":
    app.run_server(debug=True)
