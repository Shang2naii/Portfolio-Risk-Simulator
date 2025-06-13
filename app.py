import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio Risk Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def fetch_data(tickers, period="3y"):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist["Close"]
        except:
            st.warning(f"Could not fetch data for {ticker}")
    return pd.DataFrame(data).dropna()


@st.cache_data
def calculate_returns(prices):
    return prices.pct_change().dropna()


def calculate_portfolio_metrics(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - 0.04) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


def calculate_var(returns, weights, confidence_level=0.95, time_horizon=1):
    portfolio_returns = (returns * weights).sum(axis=1)

    # Historical VaR
    hist_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * np.sqrt(
        time_horizon
    )

    # Monte Carlo VaR
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    simulations = np.random.normal(mu, sigma, 10000) * np.sqrt(time_horizon)
    mc_var = np.percentile(simulations, (1 - confidence_level) * 100)

    return hist_var, mc_var


def efficient_frontier(returns, num_portfolios=1000):
    n_assets = len(returns.columns)
    results = np.zeros((3, num_portfolios))
    weights_array = np.zeros((num_portfolios, n_assets))

    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_array[i, :] = weights

        portfolio_return, portfolio_std, sharpe_ratio = calculate_portfolio_metrics(
            returns, weights
        )
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio

    return results, weights_array


def calculate_risk_attribution(returns, weights):
    cov_matrix = returns.cov() * 252
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_var)
    contrib = weights * marginal_contrib
    percent_contrib = contrib / contrib.sum() * 100

    return marginal_contrib, percent_contrib


# UI Setup
st.title("📊 Portfolio Risk Simulator")
st.markdown("*Advanced portfolio analytics and risk management*")

# Sidebar
st.sidebar.header("Portfolio Configuration")

# Preset portfolios
preset_portfolios = {
    "Tech-heavy": {
        "AAPL": 25,
        "MSFT": 20,
        "GOOGL": 15,
        "NVDA": 15,
        "TSLA": 10,
        "META": 10,
        "AMZN": 5,
    },
    "Diversified": {"SPY": 30, "QQQ": 20, "VTI": 15, "BND": 15, "GLD": 10, "VEA": 10},
    "Dividend": {
        "JNJ": 20,
        "PG": 15,
        "KO": 15,
        "PFE": 15,
        "VZ": 15,
        "T": 10,
        "XOM": 10,
    },
}

preset_choice = st.sidebar.selectbox(
    "Choose preset portfolio:", ["Custom"] + list(preset_portfolios.keys())
)

if preset_choice != "Custom":
    selected_preset = preset_portfolios[preset_choice]
    default_tickers = list(selected_preset.keys())
    default_weights = list(selected_preset.values())
else:
    default_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    default_weights = [25, 25, 25, 25]

# Portfolio inputs
num_assets = st.sidebar.slider("Number of assets:", 2, 8, len(default_tickers))

tickers = []
weights = []

for i in range(num_assets):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        ticker = st.text_input(
            f"Asset {i+1}:",
            value=default_tickers[i] if i < len(default_tickers) else "",
            key=f"ticker_{i}",
        )
        tickers.append(ticker.upper())
    with col2:
        weight = st.number_input(
            f"Weight %:",
            min_value=0.0,
            max_value=100.0,
            value=float(default_weights[i]) if i < len(default_weights) else 0.0,
            key=f"weight_{i}",
        )
        weights.append(weight)

# Validate weights
total_weight = sum(weights)
if total_weight != 100:
    st.sidebar.error(f"Weights sum to {total_weight:.1f}%. Must equal 100%.")
    st.stop()

# Settings
st.sidebar.header("Analysis Settings")
var_confidence = st.sidebar.selectbox("VaR Confidence Level:", [95, 99], index=0) / 100
time_horizon = st.sidebar.selectbox("Time Horizon (days):", [1, 5, 10, 22], index=0)
num_simulations = st.sidebar.slider("Efficient Frontier Portfolios:", 100, 5000, 1000)

# Add benchmark
benchmark = st.sidebar.selectbox("Benchmark:", ["SPY", "QQQ", "VTI"])

# Fetch data
tickers = [t for t in tickers if t]
if len(tickers) < 2:
    st.error("Please enter at least 2 valid tickers.")
    st.stop()

with st.spinner("Fetching market data..."):
    all_tickers = tickers + [benchmark]
    prices = fetch_data(all_tickers)

if prices.empty or len(prices.columns) < 2:
    st.error("Could not fetch sufficient data. Please check your tickers.")
    st.stop()

# Filter to only tickers we have data for
available_tickers = [t for t in tickers if t in prices.columns]
if len(available_tickers) != len(tickers):
    missing = set(tickers) - set(available_tickers)
    st.warning(f"Missing data for: {', '.join(missing)}")

# Adjust weights for available tickers
weights_dict = dict(zip(tickers, weights))
available_weights = [weights_dict[t] for t in available_tickers if t in weights_dict]
if available_weights:
    available_weights = np.array(available_weights) / sum(available_weights)

returns = calculate_returns(prices)

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Summary",
        "⚠️ Value at Risk",
        "🔗 Correlations",
        "📊 Efficient Frontier",
        "🎯 Risk Attribution",
        "📉 Benchmark",
    ]
)

with tab1:
    st.header("Portfolio Summary")

    col1, col2, col3 = st.columns(3)

    portfolio_return, portfolio_std, sharpe_ratio = calculate_portfolio_metrics(
        returns[available_tickers], available_weights
    )

    with col1:
        st.metric("Expected Annual Return", f"{portfolio_return:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{portfolio_std:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # Portfolio composition
    st.subheader("Portfolio Composition")
    composition_df = pd.DataFrame(
        {"Asset": available_tickers, "Weight": available_weights * 100}
    )

    fig_pie = px.pie(
        composition_df, values="Weight", names="Asset", title="Asset Allocation"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Performance chart
    st.subheader("Cumulative Returns")
    portfolio_prices = (prices[available_tickers] * available_weights).sum(axis=1)
    portfolio_cum_returns = (portfolio_prices / portfolio_prices.iloc[0] - 1) * 100

    benchmark_cum_returns = (prices[benchmark] / prices[benchmark].iloc[0] - 1) * 100

    fig_performance = go.Figure()
    fig_performance.add_trace(
        go.Scatter(
            x=portfolio_cum_returns.index,
            y=portfolio_cum_returns,
            name="Portfolio",
            line=dict(width=3),
        )
    )
    fig_performance.add_trace(
        go.Scatter(
            x=benchmark_cum_returns.index,
            y=benchmark_cum_returns,
            name=benchmark,
            line=dict(dash="dash"),
        )
    )

    fig_performance.update_layout(
        title="Portfolio vs Benchmark Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
    )
    st.plotly_chart(fig_performance, use_container_width=True)

with tab2:
    st.header("Value at Risk Analysis")

    hist_var, mc_var = calculate_var(
        returns[available_tickers], available_weights, var_confidence, time_horizon
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Historical VaR ({int(var_confidence*100)}%)", f"{hist_var:.2%}")
        st.caption(f"Maximum expected loss over {time_horizon} days")
    with col2:
        st.metric(f"Monte Carlo VaR ({int(var_confidence*100)}%)", f"{mc_var:.2%}")
        st.caption("Based on 10,000 simulations")

    # VaR interpretation
    st.info(
        f"""
    **VaR Interpretation:** With {int(var_confidence*100)}% confidence, your portfolio will not lose more than
    {abs(hist_var):.2%} over the next {time_horizon} day(s). This means there's a {int((1-var_confidence)*100)}%
    chance of losses exceeding this amount.
    """
    )

    # Distribution plot
    portfolio_returns = (returns[available_tickers] * available_weights).sum(axis=1)

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(x=portfolio_returns, nbinsx=50, name="Daily Returns", opacity=0.7)
    )
    fig_dist.add_vline(
        x=hist_var,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Historical VaR: {hist_var:.2%}",
    )
    fig_dist.add_vline(
        x=mc_var,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Monte Carlo VaR: {mc_var:.2%}",
    )

    fig_dist.update_layout(
        title="Portfolio Return Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("Asset Correlations")

    corr_matrix = returns[available_tickers].corr()

    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )
    fig_heatmap.update_layout(width=600, height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Correlation Analysis")
    st.write(
        "**High correlations (>0.7)** indicate assets move together, reducing diversification benefits."
    )
    st.write("**Low/negative correlations (<0.3)** provide better diversification.")

    # Highlight high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                )

    if high_corr_pairs:
        st.warning("**High Correlation Pairs:**")
        for asset1, asset2, corr in high_corr_pairs:
            st.write(f"• {asset1} - {asset2}: {corr:.2f}")

with tab4:
    st.header("Efficient Frontier")

    with st.spinner("Computing efficient frontier..."):
        results, weights_array = efficient_frontier(
            returns[available_tickers], num_simulations
        )

    fig_frontier = go.Figure()

    # Scatter plot of portfolios
    fig_frontier.add_trace(
        go.Scatter(
            x=results[1],
            y=results[0],
            mode="markers",
            marker=dict(
                color=results[2],
                colorscale="Viridis",
                size=4,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            name="Simulated Portfolios",
        )
    )

    # Current portfolio
    fig_frontier.add_trace(
        go.Scatter(
            x=[portfolio_std],
            y=[portfolio_return],
            mode="markers",
            marker=dict(color="red", size=15, symbol="star"),
            name="Your Portfolio",
        )
    )

    fig_frontier.update_layout(
        title="Efficient Frontier",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Expected Return",
        width=800,
        height=600,
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    # Find optimal portfolio
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_array[max_sharpe_idx]

    st.subheader("Optimal Portfolio (Max Sharpe Ratio)")
    optimal_df = pd.DataFrame(
        {
            "Asset": available_tickers,
            "Current Weight": available_weights * 100,
            "Optimal Weight": optimal_weights * 100,
        }
    )
    st.dataframe(optimal_df, use_container_width=True)

with tab5:
    st.header("Risk Attribution")

    marginal_contrib, percent_contrib = calculate_risk_attribution(
        returns[available_tickers], available_weights
    )

    risk_df = pd.DataFrame(
        {
            "Asset": available_tickers,
            "Weight (%)": available_weights * 100,
            "Risk Contribution (%)": percent_contrib,
        }
    )

    st.dataframe(risk_df, use_container_width=True)

    # Risk contribution chart
    fig_risk = px.bar(
        risk_df,
        x="Asset",
        y="Risk Contribution (%)",
        title="Risk Contribution by Asset",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.info(
        """
    **Risk Attribution** shows how much each asset contributes to overall portfolio risk.
    Assets with higher risk contribution may be candidates for position reduction.
    """
    )

with tab6:
    st.header("Benchmark Comparison")

    # Performance metrics comparison
    benchmark_returns = returns[benchmark]
    benchmark_annual_return = benchmark_returns.mean() * 252
    benchmark_std = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = (benchmark_annual_return - 0.04) / benchmark_std

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio")
        st.metric("Annual Return", f"{portfolio_return:.2%}")
        st.metric("Volatility", f"{portfolio_std:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    with col2:
        st.subheader(f"{benchmark} Benchmark")
        st.metric("Annual Return", f"{benchmark_annual_return:.2%}")
        st.metric("Volatility", f"{benchmark_std:.2%}")
        st.metric("Sharpe Ratio", f"{benchmark_sharpe:.2f}")

    # Alpha calculation
    alpha = portfolio_return - benchmark_annual_return
    st.metric("Alpha (Excess Return)", f"{alpha:.2%}", delta=f"{alpha:.2%}")

    # Rolling correlation
    portfolio_returns_daily = (returns[available_tickers] * available_weights).sum(
        axis=1
    )
    rolling_corr = portfolio_returns_daily.rolling(60).corr(benchmark_returns)

    fig_corr = go.Figure()
    fig_corr.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            name=f"60-day Rolling Correlation with {benchmark}",
        )
    )
    fig_corr.update_layout(
        title=f"Portfolio Correlation with {benchmark}",
        xaxis_title="Date",
        yaxis_title="Correlation",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Portfolio Risk Simulator | Built with Streamlit & Python</p>
        <p><small>Disclaimer: This tool is for educational purposes only. Not financial advice.</small></p>
    </div>
    """,
    unsafe_allow_html=True,
)
