import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from pypfopt import EfficientFrontier, risk_models, expected_returns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

st.title("📊 Portfolio Analysis & Prediction Dashboard")

# --- Ticker Input ---
ticker_input = st.text_input("Enter Tickers (comma-separated)", value="AAPL, MSFT, TSLA")

if ticker_input:
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    units = {}

    st.subheader("📦 Number of Shares per Ticker")
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        units[t] = cols[i].number_input(f"{t}", min_value=0, value=1, step=1)

    if st.button("📈 Run Analysis"):
        with st.spinner("Fetching data and computing portfolio..."):

            # --- Data Download ---
            start = dt.datetime(2023, 10, 1)
            end = dt.datetime.now()
            units_arr = np.array([units[t] for t in tickers])
            df = yf.download(['^GSPC'] + tickers, start, end, auto_adjust=True)
            Close = df['Close'][['^GSPC'] + tickers]
            log_returns = np.log(Close / Close.shift(1)).dropna()

            def calc_beta(df):
                m = df.iloc[:, 0].values
                return pd.Series([
                    np.cov(df.iloc[:, i].values, m)[0, 1] / np.var(m) if np.var(m) > 0 else np.nan
                    for i in range(1, df.shape[1])
                ], index=df.columns[1:], name="Beta")

            beta = calc_beta(log_returns)
            stocklist = yf.download(tickers, start, end, auto_adjust=True)
            prices = stocklist['Close'].iloc[-1][tickers].values
            value = units_arr * prices
            weights = value / value.sum()
            beta = round(beta, 2)

            portfolio = pd.DataFrame({
                'Price': prices,
                'Units': units_arr,
                'Current Value': value,
                'Weights': weights.round(2),
                'Beta': beta.values,
                'Weighted Beta': weights * beta.values
            }, index=tickers)

            sp500 = yf.download('^GSPC', start, end)
            sp500_price = sp500.Close.iloc[-1]
            portfolio['SP500 Weighted Delta (point)'] = round(portfolio['Beta'] * portfolio['Price'] / sp500_price * portfolio['Units'], 2)
            portfolio['SP500 Weighted Delta (1%)'] = round(portfolio['Beta'] * portfolio['Price'] * portfolio['Units'] * 0.01, 2)

            totals = portfolio[['Current Value', 'SP500 Weighted Delta (point)', 'SP500 Weighted Delta (1%)']].sum()
            portfolio.loc['Total'] = ['', '', *totals, '', '']

            st.dataframe(portfolio)

            # --- Timeline Plot ---
            st.subheader("📈 Portfolio vs S&P 500")
            close_prices = stocklist['Close'][tickers]
            portfolio_ts = (close_prices * units_arr).sum(axis=1)

            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(x=portfolio_ts.index, y=portfolio_ts, name="Portfolio"), secondary_y=False)
            fig1.add_trace(go.Scatter(x=sp500.index, y=sp500['Close'], name="S&P 500"), secondary_y=True)
            fig1.update_layout(title="Portfolio Value vs S&P 500", template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)

            # --- VaR & CVaR ---
            st.subheader("📉 VaR and CVaR")
            log_tfsa_returns = np.log(portfolio_ts/portfolio_ts.shift(1)).dropna()
            VaR = np.percentile(log_tfsa_returns, 5)
            CVaR = log_tfsa_returns[log_tfsa_returns <= VaR].mean()
            VaR_pct, CVaR_pct = VaR * 100, CVaR * 100

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=log_tfsa_returns * 100, nbinsx=500))
            fig2.add_vline(x=VaR_pct, line=dict(color="red", dash="dash"))
            fig2.add_vline(x=CVaR_pct, line=dict(color="darkred", dash="dot"))
            st.plotly_chart(fig2, use_container_width=True)

            # --- Sharpe Ratio ---
            st.subheader("⚖️ Sharpe Ratio")
            volatility = log_tfsa_returns.rolling(60).std()*np.sqrt(60)
            sp500_log_returns = np.log(Close['^GSPC'] / Close['^GSPC'].shift(1)).dropna()
            total_return = np.exp(sp500_log_returns.sum()) - 1
            num_years = (Close.index[-1] - Close.index[0]).days / 365
            annual_sp500_return = (1 + total_return)**(1/num_years) - 1
            Rf = annual_sp500_return / 252
            sharpe_ratio = (log_tfsa_returns.rolling(60).mean() - Rf) * 60 / volatility

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=sharpe_ratio.index, y=sharpe_ratio, name="Sharpe Ratio"))
            fig3.update_layout(title="Portfolio Sharpe Ratio", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

            # --- Portfolio Optimization ---
            st.subheader("🔧 Portfolio Optimization")
            mu = expected_returns.mean_historical_return(stocklist['Close'][tickers])
            S = risk_models.sample_cov(stocklist['Close'][tickers])
            ef = EfficientFrontier(mu, S)
            optimal_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            comparison_df = pd.DataFrame({
                'Current Weight': weights,
                'Optimal Weight': pd.Series(cleaned_weights)
            })

            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['Current Weight'], name="Current"))
            fig5.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['Optimal Weight'], name="Optimal"))
            fig5.update_layout(barmode="group", title="Weight Comparison", template="plotly_white")
            st.plotly_chart(fig5, use_container_width=True)

            # --- Monte Carlo Simulation ---
            st.subheader("🎲 Monte Carlo Simulation")
            daily_std = log_tfsa_returns.std()
            u = log_tfsa_returns.mean()
            drift = u - 0.5 * log_tfsa_returns.var()
            t_intervals = 1000
            iterations = 250
            daily_returns = np.exp(drift + daily_std * norm.ppf(np.random.rand(t_intervals, iterations)))
            s0 = portfolio_ts.iloc[-1]
            price_list = np.zeros_like(daily_returns)
            price_list[0] = s0
            for t in range(1, t_intervals):
                price_list[t] = price_list[t-1] * daily_returns[t]

            fig4 = go.Figure()
            for i in range(iterations):
                fig4.add_trace(go.Scatter(x=np.arange(t_intervals), y=price_list[:, i], line=dict(width=1), showlegend=False))
            fig4.add_trace(go.Scatter(x=np.arange(t_intervals), y=price_list.mean(axis=1), name="Average Path", line=dict(color='black', dash='dash')))
            fig4.update_layout(title="Monte Carlo Simulation", template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)

            # --- LSTM Forecast ---
            st.subheader("🤖 LSTM Forecast")
            monthly_prices = stocklist.Close.resample('ME').last()
            monthly_returns = monthly_prices.pct_change().dropna()
            weights_arr = np.array([weights[i] if i in monthly_returns.columns else 0 for i in tickers])
            portfolio_monthly_returns = monthly_returns.dot(weights_arr)

            def create_sequences(data, seq_length=12):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(X), np.array(y)

            seq_length = 12
            returns_values = portfolio_monthly_returns.values
            X, y = create_sequences(returns_values)
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
            split = int(0.8 * len(X_scaled))
            X_train, X_val = X_scaled[:split], X_scaled[split:]
            y_train, y_val = y_scaled[:split], y_scaled[split:]

            model = Sequential([
                Input(shape=(seq_length, 1)),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=100, batch_size=4, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

            last_seq = returns_values[-seq_length:]
            last_seq_scaled = scaler_X.transform(last_seq.reshape(-1, 1)).reshape(1, seq_length, 1)
            pred_scaled = model.predict(last_seq_scaled)
            predicted_return = scaler_y.inverse_transform(pred_scaled)

            st.success(f"📅 Predicted Return Next Month: **{predicted_return[0][0]:.2%}**")
