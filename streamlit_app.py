# Importing dependencies
import streamlit as st
from huggingface_hub import InferenceClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_tools import *
import yfinance as yf
import talib 
import pandas as pd
import numpy as np
from gnews import GNews
from datetime import datetime, timedelta
import plotly.express as px
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# Page config 

st.set_page_config(
    page_title="SOLiGence IEAP - Intelligent Equity Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Hugging Face setup

hf_token = "huggingface_token"
hf_model = "katanemo/Arch-Router-1.5B"

client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token)


@st.cache_resource
def load_lstm_assets(stock):

    x_scalers_lstm = joblib.load(os.path.join("IEAP models", "lstm_x_scalers.pkl"))
    y_scalers_lstm = joblib.load(os.path.join("IEAP models", "lstm_y_scalers.pkl"))
    lstm_model = joblib.load(os.path.join("IEAP models", f"{stock} lstm.pkl"))

    return x_scalers_lstm, y_scalers_lstm, lstm_model


# Sidebar navigation

st.sidebar.image("assets/soligence logo.png", use_container_width=True)
st.sidebar.title("Pages")

st.sidebar.markdown("---")

if st.sidebar.button("❓ Help"):
    st.sidebar.info(
        """
        **How to use IEAP**

        📈 **Live Prices**  
        - View candlestick charts  
        - Analyse RSI & MACD  
        - Read latest news  

        📊 **EDA**  
        - Explore feature relationships  
        - Understand technical indicators  

        📉 **Predictions**  
        - Generate forecasts  
        - View buy/sell signals  

        🤖 **AI Assistant**  
        - Ask questions about stocks, models, or the app  

        ---
        Built for decision support — not financial advice.
        """
    )

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📈 Live Prices", "📊 EDA", "🤖 AI assistant", "📉 Predictions"]
)


# Shared styling

st.markdown(
    """
    <style>
        .hero {
            padding: 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            margin-bottom: 2rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(120,120,120,0.2);
            background-color: rgba(250,250,250,0.03);
            transition: 0.2s ease-in-out;
            height: 200px;
        }
        .card:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(100,150,255,0.4);
        }
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .card-desc {
            font-size: 0.95rem;
            opacity: 0.9;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Home page

if page == "🏠 Home":

    st.image("assets/soligence logo.png", width=400)

    st.markdown(
        """
        <div class="hero">
            <h1>Intelligent Equity Analytics Platform (IEAP)</h1>
            <p style="font-size:1.05rem; margin-top:0.8rem;">
            A decision-support system designed to analyse selected NASDAQ-100 stocks using data-driven techniques.
            This platform integrates real-time market data, technical indicators, sentiment analysis, and machine learning models
            to help users interpret market behaviour and support informed trading decisions.
            </p>
            <p style="font-size:1.0rem; margin-top:0.6rem;">
            Users can explore live price movements, analyse relationships between engineered features, interact with an AI assistant,
            and generate predictive insights with buy/sell signals based on model outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Explore the Platform")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🤖 Gen AI Assistant</div>
                <div class="card-desc">
                Interact with an AI-powered assistant to ask questions about stock behaviour, model outputs,
                and system insights. Includes conversational history and contextual responses.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">📈 Live Prices</div>
                <div class="card-desc">
                View real-time stock price movements with interactive candlestick charts across multiple time horizons,
                supported by live news feeds for market context.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">📊 Exploratory Data Analysis</div>
                <div class="card-desc">
                Explore relationships between engineered features, including correlations, distributions,
                and technical indicators such as MACD and RSI.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">📉 Predictions & Signals</div>
                <div class="card-desc">
                Generate forecasts using trained machine learning models and receive actionable buy/sell/hold signals
                supported by sentiment analysis and market context.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("This homepage provides an overview of the IEAP system. Use the navigation to access each component.")


# LIVE PRICES PAGE
elif page == "📈 Live Prices":  
    
    st.title("📈 Live Prices")

    # Inputs
    stock = st.selectbox("Select stock", ["MCHP", "AMAT", "PCAR", "CEG"])
    interval = st.selectbox("Select interval", ["Daily", "3 Months", "6 Months", "1 Year"])

    # Map interval to start date
    if interval == "Daily":
        start = datetime.today() - timedelta(days=90)
    elif interval == "3 Months":
        start = datetime.today() - timedelta(days=270)
    elif interval == "6 Months":
        start = datetime.today() - timedelta(days=540)
    else:
        start = datetime.today() - timedelta(days=1095)

    start = start.strftime('%Y-%m-%d')

    # Load data
    stocks = download_stock_data([stock], start=start)

    if stock not in stocks:
        st.error(f"No data returned for {stock}.")
    else:
        df = stocks[stock]

        # Feature engineering
        df = engineer_features(df)
        df = df.reset_index()

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.22, 0.23],
            subplot_titles=(f"{stock} Candlestick", "RSI (14)", "MACD")
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price"
            ),
            row=1, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["RSI_14"],
                mode="lines",
                name="RSI"
            ),
            row=2, col=1
        )

        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["MACD"],
                mode="lines",
                name="MACD"
            ),
            row=3, col=1
        )

        # Force date labels to show under every subplot
        fig.update_xaxes(showticklabels=True, row=1, col=1)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        fig.update_xaxes(showticklabels=True, row=3, col=1)

        fig.update_layout(
            height=900,
            title=f"{stock} Price Chart with RSI & MACD",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader(f"📰 Latest News for {stock}")

        news_items = get_gnews_articles(stock, max_results=10)

        if news_items:
            for i, item in enumerate(news_items, start=1):
                title = item.get("title", "No title")
                publisher = item.get("publisher", {}).get("title", "Unknown source")
                url = item.get("url", "")
                published = item.get("published date", "No date")

                st.markdown(f"**{i}. {title}**")
                st.write(f"Source: {publisher}")
                st.write(f"Published: {published}")
                if url:
                    st.markdown(f"[Read article]({url})")
                st.markdown("---")
        else:
            st.info("No news articles found.")

        st.subheader("🤖 AI News Sentiment")

        if st.button(f"Analyse {stock} News Sentiment"):
            with st.spinner("Analysing news sentiment..."):
                try:
                    sentiment_output = get_news_sentiment(news_items, client, hf_model)
                    st.success("Sentiment analysis complete.")
                    st.write(sentiment_output)
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")

# EDA PAGE

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore descriptive statistics, stock relationships, performance over time, "
        "and technical market conditions for the selected NASDAQ-100 stocks."
    )

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------
    analysis_mode = st.radio(
        "Choose analysis mode",
        ["Single Stock Analysis", "Multi-Stock Comparison"],
        horizontal=True
    )

    period_label = st.selectbox(
        "Select lookback window",
        ["3 Months", "6 Months", "1 Year", "2 Years"],
        index=2
    )

    start_date = get_start_date_from_period(period_label)

    all_tickers = ["MCHP", "AMAT", "PCAR", "CEG"]

    if analysis_mode == "Single Stock Analysis":
        selected_tickers = [st.selectbox("Select stock", all_tickers)]
    else:
        selected_tickers = st.multiselect(
            "Select one or more stocks",
            all_tickers,
            default=all_tickers
        )

    st.markdown("### Select EDA modules")

    show_corr = st.checkbox("Correlation and relationship analysis", value=True)
    show_summary = st.checkbox("Summary statistics tables (.describe())", value=True)
    show_growth = st.checkbox("Time series growth / investment performance", value=True)
    show_technical = st.checkbox("Technical indicators / market condition analysis", value=True)

    if not selected_tickers:
        st.warning("Please select at least one stock.")
        st.stop()

    # --------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------
    stocks = download_stock_data(selected_tickers, start=start_date)

    prepared_stocks = {}

    for ticker in selected_tickers:
        if ticker not in stocks:
            continue

        df = stocks[ticker].copy()
        df = engineer_features(df)
        df = eda_features(df)
        df = df.reset_index()

        prepared_stocks[ticker] = df

    if not prepared_stocks:
        st.error("No stock data could be loaded for the selected inputs.")
        st.stop()

    # --------------------------------------------------
    # 1. Correlation and relationship analysis
    # --------------------------------------------------
    if show_corr:
        st.markdown("---")
        st.header("Correlation and Relationship Analysis")

        if len(prepared_stocks) < 2:
            st.info("Select at least two stocks to run correlation and relationship analysis.")
        else:
            returns_df = build_returns_matrix(prepared_stocks)
            corr_matrix = returns_df.corr()

            st.subheader("Return Correlation Heatmap")
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation of Daily Returns"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Relationship Scatter Plot")
            scatter_x = st.selectbox(
                "Select X-axis stock",
                list(prepared_stocks.keys()),
                key="corr_x"
            )
            scatter_y = st.selectbox(
                "Select Y-axis stock",
                list(prepared_stocks.keys()),
                index=min(1, len(prepared_stocks.keys()) - 1),
                key="corr_y"
            )

            if scatter_x != scatter_y:
                pair_df = returns_df[[scatter_x, scatter_y]].dropna().reset_index()
                fig_scatter = px.scatter(
                    pair_df,
                    x=scatter_x,
                    y=scatter_y,
                    trendline="ols",
                    title=f"Daily Return Relationship: {scatter_x} vs {scatter_y}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Choose two different stocks for the relationship scatter plot.")

    # --------------------------------------------------
    # 2. Summary statistics tables
    # --------------------------------------------------
    if show_summary:
        st.markdown("---")
        st.header("Summary Statistics")

        for ticker, df in prepared_stocks.items():
            st.subheader(f"{ticker} - Descriptive Statistics")

            summary_cols = [
                "Open", "High", "Low", "Close", "Volume",
                "RSI_14", "MACD", "Daily_Return", "Rolling_Vol_20"
            ]

            available_cols = [col for col in summary_cols if col in df.columns]
            summary_df = df[available_cols].describe().T

            st.dataframe(summary_df, use_container_width=True)

    # --------------------------------------------------
    # 3. Time series growth / investment performance
    # --------------------------------------------------
    if show_growth:
        st.markdown("---")
        st.header("Investment Growth Over Time")

        growth_df = build_normalized_growth_df(prepared_stocks)

        if growth_df.empty:
            st.info("Not enough data to calculate growth.")
        else:
            fig_growth = px.line(
                growth_df,
                x="Date",
                y="Growth_Index",
                color="Ticker",
                title="Growth of a Hypothetical £100 Investment"
            )
            fig_growth.update_layout(yaxis_title="Growth Index (Base = 100)")
            st.plotly_chart(fig_growth, use_container_width=True)

            st.caption(
                "Each stock is rebased to 100 at the starting date so performance can be compared fairly."
            )

    # --------------------------------------------------
    # 4. Technical indicators / market condition analysis
    # --------------------------------------------------
    if show_technical:
        st.markdown("---")
        st.header("Technical Indicators and Market Condition Analysis")

        tech_ticker = st.selectbox(
            "Select stock for technical analysis",
            list(prepared_stocks.keys()),
            key="tech_stock"
        )

        tech_df = prepared_stocks[tech_ticker].copy().dropna()

        if tech_df.empty:
            st.info("Not enough data to plot technical indicators.")
        else:
            interpretation = interpret_market_condition(tech_df)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RSI State", interpretation["RSI"])
            m2.metric("Trend", interpretation["Trend"])
            m3.metric("Volatility", interpretation["Volatility"])
            m4.metric("Bollinger State", interpretation["Bollinger"])

            st.subheader(f"{tech_ticker} Price with Moving Averages and Bollinger Bands")

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["Close"], name="Close"))
            fig_price.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["SMA_20"], name="SMA 20"))
            fig_price.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["SMA_50"], name="SMA 50"))
            fig_price.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["BB_Upper"], name="BB Upper"))
            fig_price.add_trace(go.Scatter(x=tech_df["Date"], y=tech_df["BB_Lower"], name="BB Lower"))
            fig_price.update_layout(title="Price, Moving Averages, and Bollinger Bands")
            st.plotly_chart(fig_price, use_container_width=True)

            st.subheader(f"{tech_ticker} RSI")
            fig_rsi = px.line(tech_df, x="Date", y="RSI_14", title="RSI (14)")
            fig_rsi.add_hline(y=70, line_dash="dash")
            fig_rsi.add_hline(y=30, line_dash="dash")
            st.plotly_chart(fig_rsi, use_container_width=True)

            st.subheader(f"{tech_ticker} MACD")
            fig_macd = px.line(tech_df, x="Date", y="MACD", title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

            st.subheader(f"{tech_ticker} Rolling Volatility")
            fig_vol = px.line(
                tech_df,
                x="Date",
                y="Rolling_Vol_20",
                title="20-Day Rolling Volatility"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

# AI ASSISTANT PAGE

elif page == "🤖 AI assistant":

    st.title("🤖 AI Assistant")
    st.write(
        "Ask questions about the application, stock analysis concepts, model outputs, "
        "or general finance topics."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello. I'm your IEAP assistant. "
                    "Ask me about the application, stock analysis, machine learning models, "
                    "or finance concepts."
                )
            }
        ]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input("Ask the assistant something...")

    if user_prompt:
        with st.chat_message("user"):
            st.write(user_prompt)

        try:
            app_context = f"""Current app context:
            - Current page: {page}
            - Available stocks: MCHP, AMAT, PCAR, CEG
            - Available pages: Home, Live Prices, EDA, AI Assistant, Predictions
            """
            reply, updated_history = chat_with_model(
                user_message=user_prompt,
                chat_history=st.session_state.chat_history,
                client=client,
                hf_model=hf_model
            )

            st.session_state.chat_history = updated_history

            with st.chat_message("assistant"):
                st.write(reply)

        except Exception as e:
            st.error(f"Model request failed: {e}")

# Predictions Page


elif page == "📉 Predictions":

    st.title("📉 Predictions")
    st.info(
        "If you have any questions about forecasts, metrics, or model behaviour, "
        "please use the AI Assistant page."
    )

    stock = st.selectbox("Select stock for prediction", ["MCHP", "AMAT", "PCAR", "CEG"])
    horizon = st.selectbox("Select forecast horizon", ["1 Day", "5 Days", "10 Days"])
    horizon_map = {"1 Day": 1, "5 Days": 5, "10 Days": 10}
    forecast_steps = horizon_map[horizon]
    model = st.selectbox(
        "Select the prediction model",
        ["XGboost Regressor", "ElasticNet Regressor", "LSTM", "MLP"]
    )

    tickers = ['MCHP', 'AMAT', 'PCAR', 'CEG']

    forecast_data = download_stock_data(tickers, start = "2026-03-01")

    sentiment_dict = get_sentiment_for_tickers(tickers)

    forecast_data = apply_sentiment_to_stocks(forecast_data, sentiment_dict)

    for ticker, df in forecast_data.items():
        df = engineer_features(df)
        df = df.dropna().copy()
        forecast_data[ticker] = df

    if model == "XGboost Regressor":

        xgb_pipeline = joblib.load(os.path.join("IEAP models", "XGboost_pipeline.pkl"))

        # 1. convert forecast_data dict into one pooled dataframe
        pooled_frames = []

        for ticker, df in forecast_data.items():
            temp = df.copy()
            temp['Target'] = temp['Close'].shift(-1)
            temp["Ticker"] = ticker
            pooled_frames.append(temp)

        master_df = pd.concat(pooled_frames, axis=0)
       

        # 2. keep only fresh data from 15-04-2026 onward
        master_df = master_df[master_df.index >="2026-04-15"].copy()

        # 3. clean any bad rows
        master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        master_df.dropna(inplace=True)

        # 4. define features and target
        features_xgb = ['Ticker', 'Volume', 'MACD', 'RSI_14', 'SD', 'Price_Momentum',
       'Vol_Shock', 'Price_Range', 'Dist_from_EMA_20', 'Close_Lag1',
       'Close_Lag3', 'Close_Lag5', 'High_Lag1', 'Low_Lag1', 'Return_1d',
       'Return_5d', 'Sentiment_Score']
        
        X_refit = master_df[features_xgb].copy()
        y_refit = master_df["Target"].copy()

        # 5. refit saved xgboost pipeline
        xgb_pipeline.fit(X_refit, y_refit)


                # Forecast selected stock
        stock_df = forecast_data[stock].copy()

        xgb_features_only = [
            'Volume', 'MACD', 'RSI_14', 'SD', 'Price_Momentum',
            'Vol_Shock', 'Price_Range', 'Dist_from_EMA_20',
            'Close_Lag1', 'Close_Lag3', 'Close_Lag5',
            'High_Lag1', 'Low_Lag1', 'Return_1d',
            'Return_5d', 'Sentiment_Score'
        ]

        recent_history = stock_df[["Close", "High", "Low", "Volume"]].tail(30).copy()

        forecast_dates = []
        forecast_prices = []

        sentiment_score = stock_df["Sentiment_Score"].iloc[-1]

        for step in range(forecast_steps):

            df_roll = engineer_features(recent_history)
            df_roll["Sentiment_Score"] = sentiment_score

            x_row = build_latest_feature_row(df_roll, xgb_features_only)
            x_row.insert(0, "Ticker", stock)

            pred_close = xgb_pipeline.predict(x_row)[0]

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

        fig_forecast, forecast_df = plot_forecast_chart(
            hist_df=stock_df,
            forecast_dates=forecast_dates,
            forecast_prices=forecast_prices,
            stock=stock
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.subheader("📋 Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)
        st.caption("Detailed forecast values including predicted " \
        "prices, implied returns, and signals.")


        st.subheader("📊 Model Evaluation Metrics")
        metrics_xgb = joblib.load("xgb_metrics.pkl")
        st.dataframe(metrics_xgb, use_container_width=True)
        st.caption("Performance metrics computed on the test dataset during model development.")

    elif model == "ElasticNet Regressor":
        
        elastic = {}

        for ticker in tickers:
            path = os.path.join("IEAP models", f"{ticker} en.pkl")
            elastic[ticker] = joblib.load(path)

        elastic_features = [
        'Volume', 'MACD', 'RSI_14', 'SD', 'Price_Momentum', 'Vol_Shock',
        'Price_Range', 'Dist_from_EMA_20', 'Sentiment_Score', 'Close_Lag1',
        'Close_Lag3', 'Close_Lag5', 'High_Lag1', 'Low_Lag1', 'Return_1d',
        'Return_5d']
    

        forecast_dates, forecast_prices = forecast_with_single_model(
            model_dict=elastic,
            forecast_data=forecast_data,
            stock=stock,
            selected_features=elastic_features,
            steps=forecast_steps
        )

        fig_forecast, forecast_df = plot_forecast_chart(
            hist_df=forecast_data[stock],
            forecast_dates=forecast_dates,
            forecast_prices=forecast_prices,
            stock=stock
        )

        st.subheader("📈 Forecast Output")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)
        st.subheader("📊 Model Evaluation Metrics")
        metrics_EN = joblib.load("elastic_net_metrics.pkl")
        st.dataframe(metrics_EN, use_container_width=True)
        st.caption("Performance metrics computed on the test dataset during model development.")

    elif model == "MLP":

        mlp = {}

        for ticker in tickers:
            path = os.path.join("IEAP models", f"{ticker} mlp.pkl")
            mlp[ticker] = joblib.load(path)

        x_scalers_mlp = joblib.load(os.path.join("IEAP models", "mlp_x_scalers.pkl"))
        y_scalers_mlp = joblib.load(os.path.join("IEAP models", "mlp_y_scalers.pkl"))

        mlp_features = [
            'Volume', 'MACD', 'RSI_14', 'SD', 'Price_Momentum', 'Vol_Shock',
            'Price_Range', 'Dist_from_EMA_20', 'Sentiment_Score', 'Close_Lag1',
            'Close_Lag3', 'Close_Lag5', 'High_Lag1', 'Low_Lag1', 'Return_1d',
            'Return_5d'
        ]

        forecast_dates, forecast_prices = forecast_with_mlp(
            model_dict=mlp,
            x_scalers=x_scalers_mlp,
            y_scalers=y_scalers_mlp,
            forecast_data=forecast_data,
            stock=stock,
            selected_features=mlp_features,
            steps=forecast_steps
        )

        fig_forecast, forecast_df = plot_forecast_chart(
            hist_df=forecast_data[stock],
            forecast_dates=forecast_dates,
            forecast_prices=forecast_prices,
            stock=stock
        )

        st.subheader("📈 MLP Forecast")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)
        st.subheader("📊 Model Evaluation Metrics")
        metrics_mlp = joblib.load("mlp_metrics.pkl")
        st.dataframe(metrics_mlp, use_container_width=True)
        st.caption("Performance metrics computed on the test dataset during model development.")

    elif model == "LSTM":

        x_scalers_lstm, y_scalers_lstm, lstm_model = load_lstm_assets(stock)

        lstm_features = [
            'Volume', 'MACD', 'RSI_14', 'SD', 'Price_Momentum', 'Vol_Shock',
            'Price_Range', 'Dist_from_EMA_20', 'Sentiment_Score', 'Close_Lag1',
            'Close_Lag3', 'Close_Lag5', 'High_Lag1', 'Low_Lag1', 'Return_1d',
            'Return_5d'
        ]

        forecast_dates, forecast_prices = forecast_with_lstm(
            model=lstm_model,
            x_scaler=x_scalers_lstm[stock],
            y_scaler=y_scalers_lstm[stock],
            forecast_data=forecast_data,
            stock=stock,
            selected_features=lstm_features,
            steps=forecast_steps,
            sequence_length=10
        )

        fig_forecast, forecast_df = plot_forecast_chart(
            hist_df=forecast_data[stock],
            forecast_dates=forecast_dates,
            forecast_prices=forecast_prices,
            stock=stock
        )
        st.subheader("📈 LSTM Forecast")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📋 Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)

        st.subheader("📊 Model Evaluation Metrics")
        metrics_lstm = joblib.load("lstm_metrics.pkl")
        st.dataframe(metrics_lstm, use_container_width=True)
        st.caption("Performance metrics computed on the test dataset during model development.")
            
        
