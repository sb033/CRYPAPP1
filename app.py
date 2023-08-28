import yfinance as yf
from prophet import Prophet
from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
from tabulate import tabulate

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    symbol = request.form['symbol']
    cur_s = request.form['conv_c']
    days = int(request.form['days'])
    attribute = request.form['attr']
    crypt = {
        "BTC":"Bitcoin",
        "ETH":"Ethereum",
        "BNB":"BNB",
        "SOL":"Solana",
        "MATIC":"Polygon MAtic",
    }
    cur_symbol = {
        "USD":"$",
        "INR":"₹",
    }
    cur_sym = cur_symbol.get(cur_s)
    coin = symbol+"-"+cur_s
    full = crypt.get(symbol)
    ticker = yf.Ticker(coin)
    data = ticker.history(period="max")
    data = data.reset_index()
    cur = data.iloc[-1]
    open_p = round(cur["Open"],4)
    close_p = round(cur["Close"],4)
    vol = cur["Volume"]
    df = data[["Date", attribute]]
    df.rename(columns={"Date": "ds", attribute: "y"}, inplace=True)
    m = Prophet(seasonality_mode="multiplicative", changepoint_range=1, changepoint_prior_scale=0.75, interval_width=0.95)
    m.fit(df)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    pred = forecast.iloc[-1]
    pred_p = round(pred["yhat"],4)
    pr_high = round(pred["yhat_upper"],4)
    pr_low = round(pred["yhat_lower"],4)
    fig = plot_plotly(m, forecast)
    graphJSON = fig.to_json()
    com = plot_components_plotly(m, forecast)
    comJSON = com.to_json()
    nf = forecast[["ds","yhat","yhat_upper","yhat_lower"]]
    nf.rename(columns={"ds":"Date", "yhat":attribute, "yhat_upper":"Upper Limit", "yhat_lower":"Lower Limit"}, inplace=True)
    nf = round(nf,2)
    nf = nf.set_index(["Date",attribute,"Upper Limit","Lower Limit"])
    return render_template('forecast.html', graphJSON=graphJSON, comJSON=comJSON, attribute=attribute, full=full, days=days, cur_sym=cur_sym, open_p=open_p, pr_high=pr_high, pr_low=pr_low, close_p=close_p, vol=vol, pred_p=pred_p, tabs=[nf[-days:].to_html()], titles=[''])