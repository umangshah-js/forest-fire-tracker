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
# from kafka import KafkaAdminClient
import requests
from prometheus_client import parser  
import threading
import redis

r = redis.Redis(host="localhost", port=6379, db=0)
system_stats_lock = threading.Lock()
system_stats = {}

def get_system_stats():
    threading.Timer(1, get_system_stats).start()
    system_stats_temp = {}
    response = requests.get("http://localhost:9308/metrics").text
    # print(response.replace("#.*\n",""))
    for item in parser.text_string_to_metric_families(response):
        if item.name == "kafka_consumergroup_lag_sum":
            for sample in item.samples:
                system_stats_temp[sample.labels['topic']] = sample.value
    redis_stats = r.memory_stats()
    system_stats_temp["num_keys"] = redis_stats["keys.count"]
    system_stats_temp["redis_memory"] = f"""{round(redis_stats["total.allocated"]/(1024*1024))} MB"""
    system_stats_temp["redis_memory_peak"] = f"""{round(redis_stats["peak.allocated"]/(1024*1024))} MB"""
    system_stats_lock.acquire()
    global system_stats
    system_stats = system_stats_temp
    system_stats_lock.release()
    # print(lags)
get_system_stats()
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'redis',
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 200
})
curr_time_stamp = 0

def get_card(header,label):
    return dbc.Card(
                [
                    # dbc.CardImg(src="/static/images/placeholder286x180.png", top=True),
                    dbc.CardBody(
                        [
                            html.H1(header, className="card-title"),
                            html.P(
                                label,
                                className="card-text",
                            ),
                            
                        ]
                    ),
                ],
                style={"width": "15rem", "margin-bottom":"10px"},
            )

def get_updated_layout(timestamp):
    stats = {}
    stats["alive"] = 0
    stats["fireline"] = 0
    stats["burning"] = 0
    stats["heating"] = 0
    stats["dead"] = 0
    stats["predicted_alive"] = 0
    if(timestamp>=0):
        with open(f"assets/{timestamp}.json") as stats_f:
            stats = json.load(stats_f) 
    stats["new_burning"] = 0
    stats["burning_error"] = 0
    stats["burning_error_pr"] = 0
    stats["alive_error"] = 0
    stats["alive_error_pr"] = 0
    stats["new_dead"] = 0
    if (timestamp>0):
        with open(f"assets/{timestamp-1}.json") as prev_stats_f:
            prev_stats = json.load(prev_stats_f)
            stats["new_dead"] = stats["dead"] - prev_stats["dead"]
            stats["new_burning"] = stats["burning"] - (prev_stats["burning"] - stats["new_dead"])
            # stats["error"] = prev_stats["burning"] -(stats["dead"] - prev_stats["dead"]) + prev_stats["fireline"] - stats["burning"]
            stats["burning_error"] = prev_stats["fireline"] - stats["new_burning"]
            stats["alive_error"] = prev_stats["predicted_alive"] - stats["alive"]
            
            if(prev_stats["alive"]!=0):
                stats["alive_error_pr"] = round((stats["alive_error"] / prev_stats["alive"])*100,2)
            if(stats["new_burning"]!=0):
                stats["burning_error_pr"] = round((stats["burning_error"] / stats["new_burning"])*100,2)
    return [
        html.H1(children='Forest Fire Tracker',className="display-3", style={"text-align":"center"}),
        html.Hr(style={"width":"95%","margin":"auto", "padding-bottom":"20px"}),
        dbc.Row([
            dbc.Col(
                html.Img(id="image", src=app.get_asset_url(f"{timestamp}.png"),style={'height':'80vh','display':'block', 'margin':'auto'}) if timestamp>=0 else dbc.Spinner(spinner_style={'display':'block', 'margin':'auto'})
                ,width="6",align="center"),
            dbc.Col([
                html.H2(children="Simulation Frame: "+str(timestamp+1)),
                html.Hr(style={"width":"90%"}),
                html.H2(children="True Values"),
                dbc.Row([
                    dbc.Col(get_card(stats['alive'], "Alive"),width=4,align="center"),
                    dbc.Col(get_card(stats['heating'], "Heating"),width=4,align="center"),
                    dbc.Col(get_card(stats['burning'], "Burning"),width=4,align="center"),
                    dbc.Col(get_card(stats['new_burning'], "Newly Burning"),width=4,align="center"),
                    dbc.Col(get_card(stats['dead'], "Dead"),width=4,align="center"),
                    dbc.Col(get_card(stats['new_dead'], "Newly Dead"),width=4,align="center"),
                ], justify='evenly'),
                html.H2(children="Predictions"),
                dbc.Row([
                    dbc.Col(get_card(stats['predicted_alive'], "Predicted Alive"),width=4,align="center"),
                    dbc.Col(get_card(stats["alive_error"], "Alive Error"),width=4,align="center"),
                    dbc.Col(get_card(f"{stats['alive_error_pr']}%", "Alive Error %"),width=4,align="center"),
                    dbc.Col(get_card(stats['fireline'], "Predicted Burning"),width=4,align="center"),
                    dbc.Col(get_card(stats["burning_error"], "Burning_error"),width=4,align="center"),
                    dbc.Col(get_card(f"{stats['burning_error_pr']}%", "Burning_error %"),width=4,align="center"),
                    # get_card(stats['burning'], "Burning"),
                    # get_card(stats['dead'], "Dead"),
                ],justify='between'),
                html.Hr(style={"width":"90%"}),
                html.H2(children="System Status"),
                dbc.Row([
                    dbc.Col(get_card(system_stats['image-preprocess'], "Preprocessing Lag"),width=4,align="center"),
                    dbc.Col(get_card(system_stats['fire-prediction'], "Prediction Lag"),width=4,align="center"),
                    dbc.Col(get_card(system_stats['wfs-events'], "Broadcast Lag"),width=4,align="center"),
                    dbc.Col(get_card(system_stats['num_keys'], "Redis Key Count"),width=4,align="center"),
                    dbc.Col(get_card(system_stats["redis_memory"], "Redis Memory Usage"), width=4,align="center"),
                    dbc.Col(get_card(system_stats["redis_memory_peak"], "Redis Peak Memory Usage"), width=4,align="center")
                ],justify='between')
        ],style={"maxHeight":"90%"})
    ])]

def serve_layout():
    session_id = str(uuid.uuid4())
    cache.set(session_id,-1)
    return html.Div(children=[
            dcc.Store(data=session_id, id='session-id'),
            html.Div(id="content",children=get_updated_layout(cache.get(session_id))),
            dcc.Interval(
                id='interval-component',
                interval=1.5*1000,  # in milliseconds
                n_intervals=0,
            )
            # dcc.Graph(id="fare_vs_age", figure=fig)
        ],style={"height":"90vh"}
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
    app.run_server(debug=False)
