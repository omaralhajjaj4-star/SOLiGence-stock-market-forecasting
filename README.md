# SOLiGence-stock-market-forecasting
A Streamlit-based intelligent equity analytics platform for selected NASDAQ-100 stocks, combining live market data, technical indicators, sentiment analysis, machine learning forecasts, and buy/sell/hold signal generation.

# SOLiGence IEAP — Intelligent Equity Analytics Platform

SOLiGence IEAP is a Streamlit-based intelligent equity analytics platform developed to support financial analysis of selected NASDAQ-100 stocks. The platform combines live market data, exploratory data analysis, technical indicators, sentiment analysis, machine learning forecasting models, and an AI assistant into one interactive dashboard.

The project was developed as part of an applied AI and data science workflow, with the aim of demonstrating how machine learning and data-driven techniques can be used to support investment analysis and decision-making. The system is designed as a decision-support platform only and does not provide financial advice or execute trades.

---

## Project Overview

The platform analyses four selected NASDAQ-100 stocks:

- AMAT — Applied Materials
- CEG — Constellation Energy
- MCHP — Microchip Technology
- PCAR — PACCAR

The application allows users to:

- View live stock price movements
- Analyse technical indicators such as RSI, MACD, moving averages, rolling volatility, and Bollinger Bands
- Explore single-stock and multi-stock relationships
- Compare historical investment growth
- Generate machine learning-based forecasts
- View buy, sell, hold, strong buy, and strong sell signals
- Analyse recent stock-related news sentiment
- Interact with an AI assistant for app and finance-related questions

---

## Features

### 1. Home Page

The home page introduces the SOLiGence IEAP platform and provides a clear overview of the main application sections:

- Live Prices
- Exploratory Data Analysis
- AI Assistant
- Predictions and Signals

---

### 2. Live Prices Page

The Live Prices page allows users to select a stock and analyse its recent market behaviour using interactive visualisations.

Key features include:

- Interactive candlestick charts
- Multiple timeframes
- RSI analysis
- MACD analysis
- Live stock-related news retrieval
- AI-powered news sentiment analysis

---

### 3. Exploratory Data Analysis Page

The EDA page supports both single-stock analysis and multi-stock comparison.

Key features include:

- Descriptive statistics tables
- Correlation heatmaps
- Daily return relationship scatter plots
- Normalised investment growth comparison
- Technical market condition analysis
- RSI, MACD, Bollinger Bands, moving averages, and rolling volatility plots

The EDA page helps users understand relationships between engineered financial features and compare stock behaviour over different time windows.

---

### 4. AI Assistant Page

The AI Assistant page allows users to ask questions about:

- The application
- Selected stocks
- Model outputs
- Technical indicators
- Forecasting behaviour
- General finance and machine learning concepts

The assistant uses contextual information from the platform and maintains short conversational history during the session.

---

### 5. Predictions Page

The Predictions page allows users to generate forecasts for selected stocks using different machine learning models.

Supported forecast horizons:

- 1 day
- 5 days
- 10 days

Supported models:

- XGBoost Regressor
- ElasticNet Regressor
- Multi-Layer Perceptron
- Long Short-Term Memory neural network

The output includes:

- Forecasted prices
- Implied returns
- Buy/sell/hold signals
- Forecast visualisation
- Model evaluation metrics

---

## Machine Learning Models

This project uses multiple regression models to forecast future stock prices or returns.

### Models Used

| Model | Purpose |
|---|---|
| ElasticNet Regressor | Linear regularised baseline model |
| XGBoost Regressor | Tree-based ensemble model for non-linear relationships |
| MLP | Neural network model for tabular financial features |
| LSTM | Sequence-based neural network model for time-series forecasting |

---

## Feature Engineering

The following features were engineered and used across the modelling pipeline:

- Volume
- MACD
- RSI 14
- Rolling standard deviation
- Price momentum
- Volume shock
- Price range
- Distance from EMA 20
- Sentiment score
- Close lag 1
- Close lag 3
- Close lag 5
- High lag 1
- Low lag 1
- 1-day return
- 5-day return

Technical indicators were created from historical OHLCV stock data and used to support both the EDA and prediction components.

---

## Signal Generation

Forecast outputs are converted into trading-style signals using implied return thresholds.

| Implied Return | Signal |
|---|---|
| Greater than 5% | Strong Buy |
| Greater than 2% | Buy |
| Between -2% and 2% | Hold |
| Less than -2% | Sell |
| Less than -5% | Strong Sell |

These signals are designed for decision-support and should not be interpreted as financial advice.

---

## Tech Stack

### Programming Language

- Python

### Application Framework

- Streamlit

### Data Collection

- yfinance
- GNews
- Finnhub API

### Data Analysis and Processing

- pandas
- NumPy

### Visualisation

- Plotly
- Plotly Express

### Technical Indicators

- TA-Lib

### Machine Learning

- Scikit-learn
- XGBoost
- TensorFlow/Keras
- joblib

### AI and NLP

- Hugging Face Inference API
- FinBERT
- LLM-based assistant functionality

---

## Project Structure

SOLiGence-IEAP/
│
├── streamlit_app.py              # Main Streamlit application
├── model_tools.py                # Helper functions for data loading, feature engineering, forecasting, sentiment, and plotting
│
├── IEAP models/                  # Saved trained models and scalers
│   ├── XGboost_pipeline.pkl
│   ├── lstm_x_scalers.pkl
│   ├── lstm_y_scalers.pkl
│   ├── mlp_x_scalers.pkl
│   ├── mlp_y_scalers.pkl
│   ├── [ticker] en.pkl
│   ├── [ticker] mlp.pkl
│   └── [ticker] lstm.pkl
│
├── assets/
│   └── soligence logo.png
│
├── xgb_metrics.pkl
├── elastic_net_metrics.pkl
├── mlp_metrics.pkl
├── lstm_metrics.pkl
│
├── notebooks/
│   ├── COM724-assessment-clustering-Omar.ipynb
│   ├── COM724-AE2-EDA-Omar.ipynb
│   └── COM724-Omar-Main.ipynb
│
├── requirements.txt
└── README.md
