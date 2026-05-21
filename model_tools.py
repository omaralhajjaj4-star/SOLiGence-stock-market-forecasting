import pandas as pd
import numpy as np
import yfinance as yf
import talib
from datetime import datetime, timedelta
from gnews import GNews
from huggingface_hub import InferenceClient
import plotly.graph_objects as go
import streamlit as st

# 1. Download latest stock data as dictionary of single-stock DataFrames
@st.cache_data(show_spinner=False)
def download_stock_data(tickers, start="2026-04-15"):
    
    stocks = {}

    for ticker in tickers:
        df = yf.download(
            tickers=ticker,
            start=start,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if not df.empty:
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            stocks[ticker] = df

    return stocks


# 2. Feature engineering for one stock DataFrame

def engineer_features(df):

    df = df.copy()

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # RSI
    df['RSI_14'] = talib.RSI(df['Close'])

    # Volatility
    df['SD'] = df['Close'].rolling(window=20).std()

    # Momentum & volume shock
    df['Price_Momentum'] = df['Close'].pct_change()
    df['Vol_Shock'] = df['Volume'].pct_change()

    # Price range
    
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    # Bolinger bands

    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)
    # EMA and EMA distance

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
    df['Dist_from_EMA_20'] = (df['Close'] - ema_20) / ema_20

    # Lag features
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['High_Lag1'] = df['High'].shift(1)
    df['Low_Lag1'] = df['Low'].shift(1)

    # Returns
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Target'] = df['Close'].shift(-1)
    return df


# 3. Fetch news sentiment for multiple tickers
@st.cache_resource
def load_finbert():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        truncation=True,
        max_length=512
    )
@st.cache_data(show_spinner=False, ttl=3600)
def get_sentiment_for_tickers(tickers, api_key = 'YOUR_FINNHUB_API_KEY', weeks_back=52):

    from transformers import pipeline
    from datetime import datetime, timedelta
    import requests

    finbert_model = load_finbert()
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(weeks=weeks_back)).strftime('%Y-%m-%d')
    sentiment_dict = {}

    for ticker in tickers:

        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={api_key}"
        scores = []

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                news_items = response.json()[:20]

                for item in news_items:
                    text = item.get('headline', '') + ". " + item.get('summary', '')

                    try:
                        result = finbert_model(text)[0]

                        pos = next(i['score'] for i in result if i['label'] == 'positive')
                        neg = next(i['score'] for i in result if i['label'] == 'negative')

                        scores.append(pos - neg)

                    except Exception:
                        continue

        except Exception:
            pass

        if len(scores) > 0:
            sentiment_score = float(np.mean(scores))
        else:
            sentiment_score = 0.0

        sentiment_dict[ticker] = sentiment_score

    return sentiment_dict


# 4. Apply sentiment score to dictionary of stock DataFrames

def apply_sentiment_to_stocks(stocks, sentiment_dict):
    
    updated_stocks = {}

    for ticker, df in stocks.items():

        df = df.copy()
        sentiment_score = sentiment_dict.get(ticker, 0.0)
        df['Sentiment_Score'] = sentiment_score

        updated_stocks[ticker] = df

    return updated_stocks


# 5. Build latest feature row from one stock DataFrame

def build_latest_feature_row(df, selected_features):

    df = df.copy()

    x_row = df[selected_features].tail(1).copy()
    x_row = x_row.replace([np.inf, -np.inf], np.nan)
    x_row = x_row.fillna(0)

    return x_row


# 6. Recursive forecast for one stock DataFrame

def recursive_forecast(model, df, selected_features, steps=10):

    df = df.copy()

    recent_history = df[['Close', 'High', 'Low', 'Volume']].tail(30).copy()

    df_feat = engineer_features(df)
    x_row = build_latest_feature_row(df_feat, selected_features)

    forecast_dates = []
    forecast_prices = []

    for step in range(steps):

        pred_close = model.predict(x_row)[0]
        next_index = recent_history.index[-1] + pd.tseries.offsets.BDay(1)

        new_row = pd.DataFrame({
            'Close': [pred_close],
            'High': [pred_close],
            'Low': [pred_close],
            'Volume': [recent_history['Volume'].iloc[-1]]
        }, index=[next_index])

        recent_history = pd.concat([recent_history, new_row])

        df_roll = engineer_features(recent_history)
        x_row = df_roll[selected_features].tail(1).copy()
        x_row = x_row.replace([np.inf, -np.inf], np.nan)
        x_row = x_row.fillna(0)

        forecast_dates.append(next_index)
        forecast_prices.append(pred_close)

    return forecast_dates, forecast_prices


# 7. Generate signal

def generate_signal(current_close, predicted_close):

    implied_return = (predicted_close / current_close) - 1

    if implied_return > 0.05:
        signal = "STRONG BUY"
    elif implied_return > 0.02:
        signal = "BUY"
    elif implied_return < -0.05:
        signal = "STRONG SELL"
    elif implied_return < -0.02:
        signal = "SELL"
    else:
        signal = "HOLD"

    return implied_return, signal


# 8. LSTM sequencer

def sequencer(X, y, seq_length):
    """Converts flat 2D arrays into 3D sequences for LSTM"""

    X_seq, y_seq = [], []

    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)

    # Helper functions for the EDA page

def get_start_date_from_period(period_label: str) -> str:
    today = datetime.today()

    if period_label == "3 Months":
        start = today - timedelta(days=90)
    elif period_label == "6 Months":
        start = today - timedelta(days=180)
    elif period_label == "1 Year":
        start = today - timedelta(days=365)
    else:
        start = today - timedelta(days=730)

    return start.strftime("%Y-%m-%d")


def eda_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Simple moving averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Rolling volatility of daily returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Rolling_Vol_20"] = df["Daily_Return"].rolling(window=20).std()

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Mid"] = rolling_mean
    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)

    return df


def build_normalized_growth_df(stock_frames: dict) -> pd.DataFrame:
    growth_frames = []

    for ticker, df in stock_frames.items():
        temp = df.copy()
        temp = temp.dropna(subset=["Close"]).copy()

        if temp.empty:
            continue

        base_price = temp["Close"].iloc[0]
        temp["Growth_Index"] = (temp["Close"] / base_price) * 100
        temp["Ticker"] = ticker

        growth_frames.append(temp[["Date", "Ticker", "Growth_Index"]])

    if not growth_frames:
        return pd.DataFrame()

    return pd.concat(growth_frames, axis=0)


def build_returns_matrix(stock_frames: dict) -> pd.DataFrame:
    returns_dict = {}

    for ticker, df in stock_frames.items():
        temp = df.copy()
        temp["Return"] = temp["Close"].pct_change()
        returns_dict[ticker] = temp.set_index("Date")["Return"]

    returns_df = pd.DataFrame(returns_dict)
    return returns_df


def build_close_matrix(stock_frames: dict) -> pd.DataFrame:
    close_dict = {}

    for ticker, df in stock_frames.items():
        close_dict[ticker] = df.set_index("Date")["Close"]

    close_df = pd.DataFrame(close_dict)
    return close_df


def interpret_market_condition(df: pd.DataFrame) -> dict:
    latest = df.dropna().iloc[-1]

    rsi = latest["RSI_14"]
    macd = latest["MACD"]
    sma_20 = latest["SMA_20"]
    sma_50 = latest["SMA_50"]
    close = latest["Close"]
    vol_20 = latest["Rolling_Vol_20"]
    bb_upper = latest["BB_Upper"]
    bb_lower = latest["BB_Lower"]

    # RSI interpretation
    if rsi >= 70:
        rsi_label = "Overbought"
    elif rsi <= 30:
        rsi_label = "Oversold"
    else:
        rsi_label = "Neutral"

    # Trend / market structure
    if close > sma_20 and sma_20 > sma_50 and macd > 0:
        trend_label = "Bullish"
    elif close < sma_20 and sma_20 < sma_50 and macd < 0:
        trend_label = "Bearish"
    else:
        trend_label = "Consolidation / Mixed"

    # Volatility interpretation
    if vol_20 < 0.015:
        vol_label = "Stable"
    elif vol_20 < 0.03:
        vol_label = "Moderately Volatile"
    else:
        vol_label = "Highly Volatile"

    # Bollinger interpretation
    if close >= bb_upper:
        bb_label = "Trading near upper Bollinger Band"
    elif close <= bb_lower:
        bb_label = "Trading near lower Bollinger Band"
    else:
        bb_label = "Trading within Bollinger range"

    return {
            "RSI": rsi_label,
            "Trend": trend_label,
            "Volatility": vol_label,
            "Bollinger": bb_label}

# Chat function

def chat_with_model(
    user_message,
    chat_history,
    client,
    hf_model="katanemo/Arch-Router-1.5B",
    app_context="",
    max_history=10,
    temperature=0.4,
    max_tokens=400
):
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string.")

    if chat_history is None:
        chat_history = []

    system_message = {
        "role": "system",
        "content": (
            "You are the AI assistant inside the SOLiGence Intelligent Equity Analytics Platform (IEAP). "
            "The platform analyses four NASDAQ-100 stocks only: MCHP, AMAT, PCAR, and CEG. "
            "The app contains these pages: Home, Live Prices, EDA, AI Assistant, and Predictions. "
            "Live Prices shows candlestick charts, RSI, MACD, and stock-specific news. "
            "EDA shows descriptive statistics, correlation analysis, normalized growth, and technical indicators. "
            "Predictions shows model forecasts, buy/sell/hold signals, and sentiment context. "
            "The forecasting models used in the platform are Elastic Net, XGBoost, LSTM, and MLP. "
            "Technical indicators include RSI, MACD, moving averages, rolling volatility, and Bollinger Bands. "
            "Use the provided app context when answering. "
            "Do not invent live data. If the user asks about current numbers, rely only on the supplied context."
        )
    }

    trimmed_history = chat_history[-max_history:] if max_history > 0 else []

    full_user_message = user_message.strip()
    if app_context:
        full_user_message = f"{app_context}\n\nUser question: {full_user_message}"

    messages = [system_message] + trimmed_history + [
        {"role": "user", "content": full_user_message}
    ]

    response = client.chat.completions.create(
        model=hf_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    assistant_reply = response.choices[0].message.content

    updated_history = chat_history + [
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": assistant_reply}
    ]

    return assistant_reply, updated_history

def get_gnews_articles(query, max_results=5):
    google_news = GNews(language='en', country='US', max_results=max_results)
    news = google_news.get_news(query)
    return news


def get_news_sentiment(news_items, client, hf_model):

    if not news_items:
        return "No news found."

    combined_text = ""

    for item in news_items[:5]:
        title = item.get("title", "")
        description = item.get("description", "")
        combined_text += f"Headline: {title}\nDescription: {description}\n\n"

    response = client.chat.completions.create(
        model=hf_model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise financial sentiment analysis assistant."
            },
            {
                "role": "user",
                "content": (
                    "Analyse the overall sentiment of the following stock news. "
                    "Classify it as Positive, Negative, or Neutral, then briefly explain why.\n\n"
                    f"{combined_text}"
                )
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message.content


def plot_forecast_chart(hist_df, forecast_dates, forecast_prices, stock):

    hist_df = hist_df.copy()

    current_close = float(hist_df["Close"].iloc[-1])

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast_Close": forecast_prices
    })

    implied_returns = []
    signals = []

    for pred_close in forecast_df["Forecast_Close"]:
        implied_return, signal = generate_signal(current_close, pred_close)
        implied_returns.append(implied_return)
        signals.append(signal)

    forecast_df["Implied_Return"] = implied_returns
    forecast_df["Signal"] = signals

    fig = go.Figure()

    # Historical close
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["Close"],
            mode="lines",
            name="Close"
        )
    )

    # SMA 20
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["SMA_20"],
            mode="lines",
            name="SMA 20"
        )
    )

    # EMA 20
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["EMA_20"],
            mode="lines",
            name="EMA 20"
        )
    )

    # Bollinger upper
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_Upper"],
            mode="lines",
            name="BB Upper",
            line=dict(dash="dash")
        )
    )

    # Bollinger lower
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df["BB_Lower"],
            mode="lines",
            name="BB Lower",
            line=dict(dash="dash")
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Close"],
            mode="lines",
            name="Forecast",
            text=forecast_df["Signal"],
            customdata=forecast_df["Implied_Return"],
            hovertemplate=(
                "Date: %{x}<br>"
                "Forecast Close: %{y:.2f}<br>"
                "Signal: %{text}<br>"
                "Implied Return: %{customdata:.2%}<extra></extra>"
            )
        )
    )
        # Map signals to colors
    color_map = {
        "STRONG BUY": "green",
        "BUY": "lightgreen",
        "HOLD": "grey",
        "SELL": "orange",
        "STRONG SELL": "red"
    }

    colors = [color_map.get(sig, "grey") for sig in forecast_df["Signal"]]

    # Signal markers
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast_Close"],
            mode="markers",
            name="Signals",
            marker=dict(
                size=10,
                color=colors
            ),
            text=forecast_df["Signal"],
            customdata=forecast_df["Implied_Return"],
            hovertemplate=(
                "Date: %{x}<br>"
                "Price: %{y:.2f}<br>"
                "Signal: %{text}<br>"
                "Return: %{customdata:.2%}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=f"{stock} Forecast with Technical Levels",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    return fig, forecast_df

def forecast_with_single_model(model_dict, forecast_data, stock, selected_features, steps=1):

    model = model_dict[stock]
    df = forecast_data[stock].copy()

    recent_history = df[["Close", "High", "Low", "Volume"]].tail(30).copy()

    sentiment_score = df["Sentiment_Score"].iloc[-1]

    forecast_dates = []
    forecast_prices = []

    for step in range(steps):

        df_roll = engineer_features(recent_history)
        df_roll["Sentiment_Score"] = sentiment_score

        x_row = build_latest_feature_row(df_roll, selected_features)

        pred_close = model.predict(x_row)[0]

        next_index = recent_history.index[-1] + pd.tseries.offsets.BDay(1)

        new_row = pd.DataFrame({
            "Close": [pred_close],
            "High": [pred_close],
            "Low": [pred_close],
            "Volume": [recent_history["Volume"].iloc[-1]]
        }, index=[next_index])

        recent_history = pd.concat([recent_history, new_row])

        forecast_dates.append(next_index)
        forecast_prices.append(pred_close)

    return forecast_dates, forecast_prices

def forecast_with_mlp(model_dict, x_scalers, y_scalers, forecast_data, stock, selected_features, steps=1):

    model = model_dict[stock]
    x_scaler = x_scalers[stock]
    y_scaler = y_scalers[stock]

    df = forecast_data[stock].copy()
    recent_history = df[["Close", "High", "Low", "Volume"]].tail(30).copy()
    sentiment_score = df["Sentiment_Score"].iloc[-1]

    forecast_dates = []
    forecast_prices = []

    expected_features = list(x_scaler.feature_names_in_)

    for step in range(steps):

        df_roll = engineer_features(recent_history)
        df_roll["Sentiment_Score"] = sentiment_score

        x_row = build_latest_feature_row(df_roll, selected_features)
        x_row = x_row.reindex(columns=expected_features, fill_value=0)

        x_row_scaled = x_scaler.transform(x_row)

        y_pred_scaled = model.predict(x_row_scaled).reshape(-1, 1)
        pred_return = y_scaler.inverse_transform(y_pred_scaled).flatten()[0]
        base_close = recent_history["Close"].iloc[-1]
        pred_close = base_close * (1 + pred_return)

        next_index = recent_history.index[-1] + pd.tseries.offsets.BDay(1)

        new_row = pd.DataFrame({
            "Close": [pred_close],
            "High": [pred_close],
            "Low": [pred_close],
            "Volume": [recent_history["Volume"].iloc[-1]]
        }, index=[next_index])

        recent_history = pd.concat([recent_history, new_row])

        forecast_dates.append(next_index)
        forecast_prices.append(pred_close)

    return forecast_dates, forecast_prices

def forecast_with_lstm(model, x_scaler, y_scaler, forecast_data, stock, selected_features, steps=1, sequence_length=10):

    df = forecast_data[stock].copy()
    history_window = max(sequence_length + 20, 30)
    recent_history = df[["Close", "High", "Low", "Volume"]].tail(history_window).copy()
    sentiment_score = df["Sentiment_Score"].iloc[-1]

    forecast_dates = []
    forecast_prices = []

    expected_features = selected_features.copy()

    if hasattr(x_scaler, "feature_names_in_"):
        expected_features = list(x_scaler.feature_names_in_)

    for step in range(steps):

        df_roll = engineer_features(recent_history)
        df_roll["Sentiment_Score"] = sentiment_score

        feature_df = df_roll[expected_features].copy()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna().copy()

        if len(feature_df) < sequence_length:
            feature_df = df_roll[expected_features].copy()
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.ffill().bfill().fillna(0)

        x_scaled = x_scaler.transform(feature_df)
        x_seq = x_scaled[-sequence_length:].reshape(1, sequence_length, len(expected_features))

        y_pred_scaled = model(x_seq, training=False).numpy()
        pred_return = y_scaler.inverse_transform(y_pred_scaled).flatten()[0]

        base_close = recent_history["Close"].iloc[-1]
        pred_close = base_close * (1 + pred_return)

        next_index = recent_history.index[-1] + pd.tseries.offsets.BDay(1)

        new_row = pd.DataFrame({
            "Close": [pred_close],
            "High": [pred_close],
            "Low": [pred_close],
            "Volume": [recent_history["Volume"].iloc[-1]]
        }, index=[next_index])

        recent_history = pd.concat([recent_history, new_row])

        forecast_dates.append(next_index)
        forecast_prices.append(pred_close)

    return forecast_dates, forecast_prices
