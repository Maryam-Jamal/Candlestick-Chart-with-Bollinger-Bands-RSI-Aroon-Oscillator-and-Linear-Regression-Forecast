#install library
pip install dash pandas yfinance plotly
pip install ta
pip install dash pandas yfinance plotly
pip install TA-Lib


import yfinance as yf
import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# Load Google data for the last 12 months
google_data = yf.download('GOOGL', period='12mo')

# Calculate 20-day Moving Average
google_data['20ma'] = google_data['Close'].rolling(window=20).mean()

# Calculate Bollinger Bands
window = 20
google_data['rolling_std'] = google_data['Close'].rolling(window=window).std()
google_data['upper_band'] = google_data['20ma'] + 2 * google_data['rolling_std']
google_data['lower_band'] = google_data['20ma'] - 2 * google_data['rolling_std']

# Calculate RSI
window = 14  # Adjust the RSI window size
delta = google_data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=window).mean()
avg_loss = loss.rolling(window=window).mean()
rs = avg_gain / avg_loss
google_data['rsi'] = 100 - (100 / (1 + rs))

# Calculate Aroon Up and Aroon Down
aroon_window = 25  # Adjust the Aroon window size
google_data['aroon_up'] = 100 * google_data['High'].rolling(window=aroon_window).apply(lambda x: x.tolist().index(max(x)) / (aroon_window - 1))
google_data['aroon_down'] = 100 * google_data['Low'].rolling(window=aroon_window).apply(lambda x: x.tolist().index(min(x)) / (aroon_window - 1))

# Calculate Linear Regression Forecast
lr_window = 10  # Adjust the window size for linear regression
lr_model = LinearRegression()
lr_x = pd.Series(range(len(google_data['Close'])), index=google_data.index).values.reshape(-1, 1)
lr_model.fit(lr_x[-lr_window:], google_data['Close'][-lr_window:])
lr_forecast = lr_model.predict(lr_x)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Google Shares Candlestick Chart with Bollinger Bands, RSI, Aroon Oscillator, and Linear Regression Forecast"),
    dcc.Graph(
        id='candlestick-chart',
        figure={
            'data': [
                go.Candlestick(
                    x=google_data.index,
                    open=google_data['Open'],
                    high=google_data['High'],
                    low=google_data['Low'],
                    close=google_data['Close'],
                    name='Google Shares'
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['20ma'],
                    mode='lines',
                    name='20-day Moving Average',
                    line=dict(color='orange')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['upper_band'],
                    mode='lines',
                    name='Upper Bollinger Band',
                    line=dict(color='red')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['lower_band'],
                    mode='lines',
                    name='Lower Bollinger Band',
                    line=dict(color='green')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['aroon_up'],
                    mode='lines',
                    name='Aroon Up',
                    line=dict(color='blue')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=google_data['aroon_down'],
                    mode='lines',
                    name='Aroon Down',
                    line=dict(color='pink')
                ),
                go.Scatter(
                    x=google_data.index,
                    y=lr_forecast,
                    mode='lines',
                    name='Linear Regression Forecast',
                    line=dict(color='brown')
                ),
            ],
            'layout': {
                'title': 'Google Shares Candlestick Chart with Bollinger Bands, RSI, Aroon Oscillator, and Linear Regression Forecast',
                'xaxis': {'rangeslider': {'visible': False}, 'type': 'category'},
                'yaxis': {'title': 'Price'},
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
