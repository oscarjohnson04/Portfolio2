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
from fredapi import Fred
import requests

st.set_page_config(layout="wide")

st.title("📊 Portfolio Analysis Dashboard")

st.sidebar.header("Benchmark Settings (Used in portfolio time series)")
benchmark_options = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Dow Jones': '^DJI',
    'RUSSELL 2000': '^RUT',
    'TSX': '^GSPTSE',
    'NIKKEI 225': '^N225',
    'HANG SENG': '^HSI',
    'NIFTY 50': '^NSEI',
    'EURO STOXX 600': '^STOXX',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'CAC 40': '^FCHI'
}
benchmark_name = st.sidebar.selectbox("Choose Benchmark:", list(benchmark_options.keys()))
benchmark_ticker = benchmark_options[benchmark_name]

@st.cache_data(show_spinner=True, ttl=60 * 15)
def fetch_news(query: str, page_size: int, sort_by: str, use_dates: bool, from_date, to_date):
    """
    Fetch news from NewsAPI's /everything endpoint. Optionally filter by date.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": int(page_size),
        "sortBy": sort_by,
        "apiKey": NEWS_API_KEY
    }
    if use_dates and from_date:
        params["from"] = str(from_date)
    if use_dates and to_date:
        params["to"] = str(to_date)

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "ok":
            return [], f"NewsAPI error: {payload.get('message', 'Unknown error')}"
        return payload.get("articles", []), None
    except Exception as e:
        return [], f"Request failed: {e}"

@st.cache_data(show_spinner=False)
def fetch_multiple_latest_series(series_ids: dict, start: dt.date, end: dt.date) -> dict:
    """Fetch the last available value for each series id in series_ids."""
    out = {}
    for label, code in series_ids.items():
        try:
            s = fred.get_series(code, start, end)
            if len(s) == 0:
                out[label] = np.nan
            else:
                out[label] = float(s.iloc[-1])
        except Exception:
            out[label] = np.nan
    return out


fred = Fred(api_key='00edddc751dd47fb05bd7483df1ed0a3')

start1 = dt.datetime(2015, 1, 1)
end = dt.datetime.now()



tab1, tab2 = st.tabs(["Portfolio Analysis", "News"])
# --- Ticker Input ---
with tab1:

    start = st.date_input("First date", value=today, min_value=dt.date(2000, 1, 1), key="yc_d1")
    
    ticker_input = st.text_input("Enter Tickers (comma-separated)", value="AAPL, MSFT, TSLA")

    if ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        units = {}

        st.subheader("Number of Shares per Ticker")
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            units[t] = cols[i].number_input(f"{t}", min_value=0, value=1, step=1)

        if st.button("📈 Run Analysis"):
            with st.spinner("Fetching data and computing portfolio..."):

            # --- Data Download ---
                sidebar_series_ids = {
                    "Real GDP (Billions, chained 2017$)": "GDPC1",
                    "Unemployment Rate": "UNRATE",
                    "Core Median CPI": "MEDCPIM158SFRBCLE",
                    "Debt/GDP Ratio": "GFDEGDQ188S",
                    "Fed Target Upper Bound": "DFEDTARU",
                    "Effective Fed Funds Rate": "DFF",
                    "3M T-Bill Yield": "DGS3MO",
                    "10Y Treasury Yield": "DGS10",
                    "30Y Treasury Yield": "DGS30",
                    "Moody's AAA Corp Yield": "DAAA",
                    "VIX": "VIXCLS",
                    "US Econ Policy Uncertainty": "USEPUINDXD",
                    "Global Econ Policy Uncertainty": "GEPUCURRENT"
                }

                st.sidebar.title("Latest US Macro Data")
                latest_data = fetch_multiple_latest_series(sidebar_series_ids, start1, end)
                for label, value in latest_data.items():
                    suffix = "%" if any(k in label.lower() for k in ["rate", "yield", "cpi", "uncertainty"]) else ""
                    prefix = "$" if "gdp" in label.lower() else ""
                    if np.isnan(value):
                        st.sidebar.metric(label, "N/A")
                    else:
                        st.sidebar.metric(label, f"{prefix}{value:,.2f}{suffix}")
                
                units_arr = np.array([units[t] for t in tickers])
                df = yf.download(['^GSPC'] + tickers, start, end, multi_level_index = False)
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
    
                # Construct portfolio DataFrame
                portfolio = pd.DataFrame({
                    'Price': prices,
                    'Units': units_arr,
                    'Current Value': value,
                    'Weights': weights,
                    'Beta': beta.values,
                    'Weighted Beta': weights * beta.values
                }, index=tickers)
    
                # Download SP500 data
                sp500 = yf.download('^GSPC', start, end)
                sp500_price = sp500['Close'].iloc[-1]
    
                # Compute SP500 weighted deltas
                portfolio['SP500 Weighted Delta (point)'] = (portfolio['Beta'].astype(float)* portfolio['Price'].astype(float)* portfolio['Units'].astype(float)) / float(sp500_price)
    
                portfolio['SP500 Weighted Delta (point)'] = portfolio['SP500 Weighted Delta (point)'].round(2)
                portfolio['SP500 Weighted Delta (1%)'] = portfolio['Beta'] * portfolio['Price'] * portfolio['Units'] * 0.01
    
                # Round all numeric columns to 2 decimals
                portfolio = portfolio.applymap(lambda x: round(x, 2) if isinstance(x, (float, int)) else x)
    
                # Compute totals
                totals = portfolio[['Current Value', 'SP500 Weighted Delta (point)', 'SP500 Weighted Delta (1%)']].sum()
                portfolio.loc['Total'] = {
                    'Price': '',
                    'Units': '',
                    'Current Value': round(totals['Current Value'], 2),
                    'Weights': '',
                    'Beta': '',
                    'Weighted Beta': '',
                    'SP500 Weighted Delta (point)': round(totals['SP500 Weighted Delta (point)'], 2),
                    'SP500 Weighted Delta (1%)': round(totals['SP500 Weighted Delta (1%)'], 2),
                }
    
                # Display
                st.subheader("Portfolio Dashboard")
                st.dataframe(portfolio)
    
                sector_map = {}
                for t in tickers:
                    try:
                        info = yf.Ticker(t).info
                        sector = info.get('sector', 'Unknown')
                        sector_map[t] = sector
                    except:
                        sector_map[t] = 'Unknown'
    
                sector_df = pd.Series({sector_map[t]: value[i] for i, t in enumerate(tickers)})
                sector_grouped = sector_df.groupby(sector_df.index).sum()
    
                ##st.subheader("Sector Allocation")
                fig_sector = go.Figure(data=[go.Pie(labels=sector_grouped.index, values=sector_grouped)])
                ##st.plotly_chart(fig_sector, use_container_width=True)
    
                # Compute monthly returns
                monthly_prices = stocklist['Close'][tickers].resample('M').last()
                monthly_returns = monthly_prices.pct_change().dropna()
    
                # Convert your existing sector_map (dict) into a pandas Series aligned to columns
                sector_series = pd.Series(sector_map).reindex(monthly_returns.columns).fillna("Unknown")
    
                # Group monthly returns by sector
                sector_returns = monthly_returns.groupby(sector_series, axis=1).mean()
    
                # Show sector-based average returns (optional table or bar chart)
                ##st.subheader("Average Monthly Returns by Sector")
                fig_bar = go.Figure(data=[
                    go.Bar(x=sector_returns.columns, y=sector_returns.mean() * 100)])
                fig_bar.update_layout(xaxis_title="Sector", yaxis_title="Return (%)")
                ##st.plotly_chart(fig_bar, use_container_width=True)
    
                col1, col2 = st.columns(2)
    
                with col1:
                    st.subheader("Sector Allocation")
                    fig_sector = go.Figure(
                        data=[go.Pie(labels=sector_grouped.index, values=sector_grouped)])
                    st.plotly_chart(fig_sector, use_container_width=True)
    
                with col2:
                    st.subheader("Average Monthly Return by Sector (%)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                
                # --- Timeline Plot ---
                st.subheader(f"Portfolio vs {benchmark_name} (Indexed to 100)")
                
                close_prices = stocklist['Close'][tickers]
                benchmark_data = yf.download(benchmark_ticker, start, end, multi_level_index=False)
                
                # Portfolio total value over time
                portfolio_ts = (close_prices * units_arr).sum(axis=1).dropna()
                
                # Align dates (use only dates present in BOTH series)
                common_dates = portfolio_ts.index.intersection(benchmark_data.index)
                portfolio_ts = portfolio_ts.loc[common_dates]
                index_ts = benchmark_data.loc[common_dates, 'Close'].dropna()
                common_dates = portfolio_ts.index.intersection(index_ts.index)
                portfolio_ts = portfolio_ts.loc[common_dates]
                index_ts = index_ts.loc[common_dates]
                
                # Rebase both to 100 at the first common date
                portfolio_idx = 100 * (portfolio_ts / portfolio_ts.iloc[0])
                index_idx = 100 * (index_ts / index_ts.iloc[0])
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=portfolio_idx.index, y=portfolio_idx, name="Portfolio"))
                fig1.add_trace(go.Scatter(x=index_idx.index, y=index_idx, name=benchmark_name))
                fig1.update_layout(
                    title=f"Portfolio vs {benchmark_name} — Indexed (Start = 100)",
                    xaxis_title="Date",
                    yaxis_title="Index (Start = 100)",
                    template="plotly_white",
                )
                st.plotly_chart(fig1, use_container_width=True)

                # st.subheader(f"Portfolio vs {benchmark_name}")
                # close_prices = stocklist['Close'][tickers]
                # benchmark_data = yf.download(benchmark_ticker, start, end, multi_level_index = False)
                # portfolio_ts = (close_prices * units_arr).sum(axis=1)
    
                # fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                # fig1.add_trace(go.Scatter(x=portfolio_ts.index, y=portfolio_ts, name="Portfolio"), secondary_y=False)
                # fig1.add_trace(go.Scatter(x=benchmark_data.index, y=benchmark_data['Close'], name=benchmark_name), secondary_y=True)
                # fig1.update_layout(title=f"Portfolio Value vs {benchmark_name}", template='plotly_white')
                # st.plotly_chart(fig1, use_container_width=True)
    
                daily_change = np.log(portfolio_ts/portfolio_ts.shift(1)).dropna()
                daily_mean_change = daily_change.mean()
                monthly_mean_change = np.exp(daily_mean_change * 21) - 1  
                yearly_mean_change  = np.exp(daily_mean_change * 252) - 1 
                st.write(f"• Mean Daily Log Return: {daily_mean_change * 100:.2f}%")
                st.write(f"• Mean Monthly Log Return: {monthly_mean_change * 100:.2f}%")
                st.write(f"• Mean Yearly Log Return: {yearly_mean_change * 100:.2f}%")
    
                dividends = {}
                for t in tickers:
                    try:
                        info = yf.Ticker(t).info
                        div_yield = info.get('dividendYield', 0)
                        div_amount = info.get('dividendRate', 0)
    
            # Normalize yield if needed
                        if div_yield:
                            div_yield = div_yield / 100 if div_yield > 0 else div_yield
                        else:
                            div_yield = 0
                            
                        total_dividend = div_amount * units.get(t, 0) if div_amount else 0
            # Store both
                        dividends[t] = {
                            'Dividend Yield': div_yield,
                            'Dividend Amount ($)': div_amount if div_amount else 0,
                            'Total Dividend ($)': total_dividend
                        }
    
                    except:
                        dividends[t] = {
                            'Dividend Yield': 0,
                            'Dividend Amount ($)': 0,
                            'Total Dividend ($)': 0
                        }
    
    # Convert to DataFrame
                div_df = pd.DataFrame.from_dict(dividends, orient='index')
    
                fig_div = go.Figure(go.Bar(
                    x=div_df.index,
                    y=div_df['Total Dividend ($)'],
                    text=[f"${val:.2f}" for val in div_df['Total Dividend ($)']],
                    textposition='outside',
                    marker_color='mediumseagreen'
                ))
    
                fig_div.update_layout(
                    xaxis_title="Ticker",
                    yaxis_title="Total Dividend ($)",
                    uniformtext_minsize=8,
                    uniformtext_mode='hide',
                    template='plotly_white'
                )
    
                col5, col6 = st.columns(2)
    
                with col5:
                    st.subheader("Dividend Summary")
                    st.dataframe(div_df.style.format({
                        "Dividend Yield": "{:.2%}",
                        "Dividend Amount ($)": "${:.2f}",
                        "Total Dividend ($)": "${:.2f}"
                    }))
                    
                with col6:
                    st.subheader("Total Dividend Income per Ticker")
                    st.plotly_chart(fig_div, use_container_width=True)
    
                st.subheader("Company Financials")
    
                financial_data = {}
                
                for t in tickers:
                    try:
                        info = yf.Ticker(t).info
                        financial_data[t] = {
                            "Market Cap ($)": info.get("marketCap", np.nan),
                            "Trailing EPS": info.get("trailingEps", np.nan),
                            "Forward EPS": info.get("forwardEps", np.nan),
                            "PE Ratio": info.get("trailingPE", np.nan),
                            "Return On Equity": info.get("returnOnEquity", np.nan),
                            "YOY Earnings Growth (%)": info.get("earningsGrowth", np.nan),
                            "YOY Revenue Growth (%)": info.get("revenueGrowth", np.nan),
                            "Total Revenue ($)": info.get("totalRevenue", np.nan),
                            "Gross Profits ($)": info.get("grossProfits", np.nan),
                            "Total Debt ($)": info.get("totalDebt", np.nan)
                        }
                        
                    except:
                        financial_data[t] = {
                            "Market Cap ($)": np.nan,
                            "Trailing EPS": np.nan,
                            "Forward EPS": np.nan,
                            "PE Ratio": np.nan,
                            "Return On Equity": np.nan,
                            "YOY Earnings Growth (%)": np.nan,
                            "YOY Revenue Growth (%)": np.nan,
                            "Total Revenue ($)": np.nan,
                            "Gross Profits ($)": np.nan,
                            "Total Debt ($)": np.nan
                        }
    
    # Convert to DataFrame and format
                fin_df = pd.DataFrame.from_dict(financial_data, orient="index")
                fin_df["YOY Earnings Growth (%)"] = fin_df["YOY Earnings Growth (%)"] * 100
                fin_df["YOY Revenue Growth (%)"] = fin_df["YOY Revenue Growth (%)"] * 100
                st.dataframe(
                    fin_df.style.format({
                        "Market Cap ($)": "${:,.0f}",
                        "Trailing EPS": "{:.2f}",
                        "Forward EPS": "{:.2f}",
                        "PE Ratio": "{:.2f}",
                        "Return On Equity": "{:.2f}",
                        "YOY Earnings Growth (%)": "{:.2f}%",
                        "YOY Revenue Growth (%)": "{:.2f}%",
                        "Total Revenue ($)": "${:,.0f}",
                        "Gross Profits ($)": "${:,.0f}",
                        "Total Debt ($)": "${:,.0f}"
                    })
                )
    
    
                st.subheader("Correlation Matrix (Returns)")
                with st.expander("ℹ️ What is a correlation matrix?"):
                    st.write("A correlation matrix displays how correlated each of your assets are to each other")
                correlation = log_returns[tickers].corr()
                fig_corr = go.Figure(data=go.Heatmap(z=correlation.values,
                                                     x=correlation.columns,
                                                     y=correlation.columns,
                                                     colorscale='RdBu', zmin=-1, zmax=1))
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # --- VaR & CVaR ---
                st.subheader("VaR, CVaR & Daily Returns")
                with st.expander("ℹ️ What is VaR and CVaR?"):
                    st.write("VaR (Value at risk) of 5% is how much your portfolio will have to lose to be in the 5% of worst daily returns")
                    st.write("CVaR (Conditional Value at risk) of 5% is how much your portfolio loses on average when it crosses the 5% VaR threshold")
                log_tfsa_returns = np.log(portfolio_ts/portfolio_ts.shift(1)).dropna()
                VaR = np.percentile(log_tfsa_returns, 5)
                CVaR = log_tfsa_returns[log_tfsa_returns <= VaR].mean()
                VaR_pct, CVaR_pct = VaR * 100, CVaR * 100
    
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=log_tfsa_returns * 100, nbinsx=500, showlegend=False))
                fig2.add_vline(x=VaR_pct, line=dict(color="red", dash="dash"))
                fig2.add_vline(x=CVaR_pct, line=dict(color="darkred", dash="dot"))
                fig2.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="95% VaR"
                ))
                fig2.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="darkred", dash="dot"),
                    name="95% CVaR"
                ))
                st.plotly_chart(fig2, use_container_width=True)
    
                st.write(f"95% VaR: {VaR_pct:.2f}%")
                st.write(f"95% CVaR: {CVaR_pct:.2f}%")
    
                def compute_drawdown(series):
                    cumulative = (1 + series).cumprod()
                    peak = cumulative.cummax()
                    drawdown = (cumulative - peak) / peak
                    return drawdown
    
                drawdowns = compute_drawdown(log_tfsa_returns)
                st.subheader("Drawdowns Time Series")
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, name="Drawdowns"))
                fig5.update_layout(template='plotly_white')
                st.plotly_chart(fig5, use_container_width=True)
                st.write(f"Max Drawdown: {drawdowns.min()*100:.2f}%")
    
                # --- Sharpe Ratio ---
                st.subheader("Sharpe Ratio")
                with st.expander("ℹ️ What is Sharpe Ratio?"):
                    st.write("The Sharpe Ratio is the average return earned in excess of the risk-free rate per unit of volatility.")
                    st.write("The risk-free rate of return used is the returns of the S&P 500")
                volatility = log_tfsa_returns.rolling(60).std()*np.sqrt(60)
                sp500_log_returns = np.log(Close['^GSPC'] / Close['^GSPC'].shift(1)).dropna()
                total_return = np.exp(sp500_log_returns.sum()) - 1
                num_years = (Close.index[-1] - Close.index[0]).days / 365
                annual_sp500_return = (1 + total_return)**(1/num_years) - 1
                Rf = annual_sp500_return / 252
                sharpe_ratio = (log_tfsa_returns.rolling(60).mean() - Rf) * 60 / volatility
    
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=sharpe_ratio.index, y=sharpe_ratio, name="Sharpe Ratio"))
                fig3.update_layout(template="plotly_white")
                st.plotly_chart(fig3, use_container_width=True)
    
                # --- Portfolio Optimization ---
                #st.subheader("Portfolio Optimization")
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
                #st.plotly_chart(fig5, use_container_width=True)
    
                current_value = value.sum()
                opt_val = current_value * pd.Series(cleaned_weights)
                opt_units = opt_val / prices
                rebalance_units = opt_units - units_arr
                rebalance_df = pd.DataFrame({'Units needed to reach optimal portfolio': rebalance_units.round(2)}, index=tickers)
                #st.subheader("Suggested Rebalancing")
                #st.dataframe(rebalance_df)
    
                col3, col4 = st.columns(2)
    
                with col3:
                    st.subheader("Portfolio Optimization")
                    st.plotly_chart(fig5, use_container_width=True)
    
                with col4:
                    st.subheader("Suggested Rebalancing")
                    st.dataframe(rebalance_df)
    
                # --- Monte Carlo Simulation ---
                st.subheader("Monte Carlo Simulation")
                daily_std = log_tfsa_returns.std()
                u = log_tfsa_returns.mean()
                drift = u - 0.5 * log_tfsa_returns.var()
                t_intervals = 1000
                iterations = 125
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
                fig4.update_layout(template="plotly_white")
                st.plotly_chart(fig4, use_container_width=True)
    
                # --- LSTM Forecast ---
                st.subheader("LSTM Forecast")
                monthly_prices = stocklist.Close.resample('ME').last()
                monthly_returns = monthly_prices.pct_change().dropna()
                weights_series = pd.Series(weights, index=tickers)
                weights_arr = np.array([weights_series[i] if i in monthly_returns.columns else 0 for i in tickers])
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
    
                st.success(f"Predicted Return Next Month: **{predicted_return[0][0]:.2%}**")

with tab2:
    NEWS_API_KEY = "80f3080a10da4d91809c5e53cf0d9828"
    colq1, colq2, colq3 = st.columns([2, 1, 1])
    with colq1:
        query = st.text_input("Search query", value="US Treasury yields OR bond market OR Federal Reserve")
    with colq2:
        page_size = st.number_input("Articles to show", min_value=3, max_value=30, value=10, step=1)
    with colq3:
        sort_by = st.selectbox("Sort by", ["Date", "Relevancy", "Popularity"], index=0)

    use_dates = st.toggle("Filter by date range", value=False)
    if use_dates:
        c1, c2 = st.columns(2)
        with c1:
            from_date = st.date_input("From date", value=dt.date.today() - dt.timedelta(days=30), key="news_from")
        with c2:
            to_date = st.date_input("To date", value=dt.date.today(), key="news_to")
        if from_date > to_date:
            st.warning("'From date' must be on or before 'To date'.")
    else:
        from_date = None
        to_date = None

    articles, err = fetch_news(query, page_size, sort_by, use_dates, from_date, to_date)

    if err:
        st.error(err)
    elif not articles:
        st.info("No articles found. Try adjusting your query or date range.")
    else:
        compact = st.toggle("Compact view", value=False)
        for a in articles:
            title = a.get("title") or "Untitled"
            url = a.get("url") or ""
            source = (a.get("source") or {}).get("name") or "Unknown source"
            published = (a.get("Date") or "")[:10]
            desc = a.get("description") or ""
            thumb = a.get("urlToImage")

            if compact:
                st.markdown(f"- **[{title}]({url})** — {source} · {published}")
            else:
                with st.container(border=True):
                    if thumb:
                        colA, colB = st.columns([1, 3])
                        with colA:
                            st.image(thumb, use_container_width=True)
                        with colB:
                            st.markdown(f"### [{title}]({url})")
                            st.caption(f"{source} · {published}")
                            if desc:
                                st.write(desc)
                    else:
                        st.markdown(f"### [{title}]({url})")
                        st.caption(f"{source} · {published}")
                        if desc:
                            st.write(desc)
