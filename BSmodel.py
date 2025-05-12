import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp 
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
import yfinance as yf
from autograd import grad
from sympy.stats import Normal, cdf

def bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    
    return call


def bsm_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
    
    
    return put


S1,K1,T1,r1,sigma1 = sp.symbols('S K T r sigma')

N = Normal('x', 0, 1)

d1 = (sp.ln(S1/K1) + (r1 + 0.5 * sigma1 ** 2) * T1)/(sigma1 * sp.sqrt(T1))
d2 = d1 - sigma1 * sp.sqrt(T1)  

call = S1 * cdf(N)(d1) - K1 * sp.exp(-r1 * T1) * cdf(N)(d2)
put = (K1 * sp.exp(-r1 * T1) * cdf(N)(-d2) - S1 * cdf(N)(-d1))

delta_call_expr = sp.diff(call, S1)
delta_put_expr = sp.diff(put, S1)
gamma_expr = sp.diff(delta_call_expr, S1)
theta_call_expr = sp.diff(call, T1)
theta_put_expr = sp.diff(put, T1)
vega_expr = sp.diff(call, sigma1)
rho_call_expr = sp.diff(call, r1)
rho_put_expr = sp.diff(put, r1)








def get_option_data(ticker1):    
    ticker = yf.Ticker(ticker1)
    expirations = ticker.options
    opt_date = expirations[2]   
    opt_chain = ticker.option_chain(opt_date)   
    calls = opt_chain.calls
    calls = calls[(calls["impliedVolatility"] > 0.01)]
    puts = opt_chain.puts   
    puts = puts[(puts["impliedVolatility"] > 0.01)]
    return calls
           
           
           
def get_option_data(ticker1):    
    ticker = yf.Ticker(ticker1)
    expirations = ticker.options
    opt_date = expirations[2]   
    opt_chain = ticker.option_chain(opt_date)   
    calls = opt_chain.calls
    calls = calls[(calls["impliedVolatility"] > 0.01)]
    puts = opt_chain.puts   
    puts = puts[(puts["impliedVolatility"] > 0.01)]
    return puts
           
def expiration_date(contractSymbol):
    date = contractSymbol[4:10]
    year = '20' + date[0:2]
    month = date[2:4]
    day = date[4:6]
    return f"{year}-{month}-{day}"





st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")
st.title("Black-Scholes Option Pricing Model")



S = float(st.number_input("Stock Price (S)", min_value=0.0, value=100.0))
K = float(st.number_input("Strike Price (K)", min_value=0.0, value=100.0))
T = float(st.number_input("Time to Expiration (T in years)", min_value=0.0, value=1.0))
r = float(st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, format="%.4f"))
sigma = float(st.number_input("Volatility (σ)", min_value=0.0, value=0.2, format="%.4f"))
ticker1 = str(st.text_input("Ticker Symbol",))

values = {
    S1: S,
    K1: K,
    T1: T,
    r1: r,
    sigma1: sigma
}       
if st.button("Calculate Option Prices"):
   
    call_price = bsm_call(S, K, T, r, sigma)
    put_price = bsm_put(S, K, T, r, sigma)
    

    st.subheader("Option Prices")
    st.write(f"**Call Option Price:** ${call_price:.2f}")
    st.write(f"**Put Option Price:** ${put_price:.2f}")
    
    

    delta_call = delta_call_expr.evalf(subs=values)
    delta_put = delta_put_expr.evalf(subs=values)
    gamma = gamma_expr.evalf(subs=values)
    theta_call = theta_call_expr.evalf(subs=values)
    theta_put = theta_put_expr.evalf(subs=values)
    vega = vega_expr.evalf(subs=values)
    rho_call = rho_call_expr.evalf(subs=values)
    rho_put = rho_put_expr.evalf(subs=values)

    st.subheader("Option Greeks")
    st.markdown(f"- **Delta (Call):** {delta_call:.4f}")
    st.markdown(f"- **Delta (Put):** {delta_put:.4f}")
    st.markdown(f"- **Gamma:** {gamma:.4f}")
    st.markdown(f"- **Theta (Call):** {theta_call:.4f}")
    st.markdown(f"- **Theta (Put):** {theta_put:.4f}")
    st.markdown(f"- **Vega:** {vega:.4f}")
    st.markdown(f"- **Rho (Call):** {rho_call:.4f}")
    st.markdown(f"- **Rho (Put):** {rho_put:.4f}")

    st.subheader("Call & Put Prices vs. Stock Price")

    S_range = np.linspace(0.5 * S, 2 * S, 100)

    call_prices = np.vectorize(bsm_call)(S_range, K, T, r, sigma)
    put_prices = np.vectorize(bsm_put)(S_range, K, T, r, sigma)

    fig_price_plot, ax_price_plot = plt.subplots(figsize=(10, 6))
    ax_price_plot.plot(S_range, call_prices, label="Call Price", color='green')
    ax_price_plot.plot(S_range, put_prices, label="Put Price", color='red')
    ax_price_plot.set_title("Option Prices vs. Stock Price")
    ax_price_plot.set_xlabel("Stock Price (S)")
    ax_price_plot.set_ylabel("Option Price")
    ax_price_plot.legend()
    ax_price_plot.grid(True)
    st.pyplot(fig_price_plot)
    
    
    calls = get_option_data(ticker1)
    puts = get_option_data(ticker1)
    expirations = expiration_date(ticker1)
    
    calls["expirations"] = calls["contractSymbol"].apply(expiration_date)
    puts["expirations"] = puts["contractSymbol"].apply(expiration_date)

    calls["timeToExpiration"] = (pd.to_datetime(calls["expirations"]) - pd.to_datetime("today")).dt.days /365
    puts["timeToExpiration"] = (pd.to_datetime(puts["expirations"]) - pd.to_datetime("today")).dt.days /365      

    
    
    implied_volatilities = []
    in_the_money = []
    time_to_expiration = []
    S = yf.download("AAPL", period="1d")["Close"].iloc[0]
    for i, row in calls.iterrows():
        K = row["strike"]
        T = row["timeToExpiration"]
        market_price = row["lastPrice"]
        implied_volatility = sigma
    
        implied_volatilities.append(implied_volatility)
        in_the_money.append(S/K)
        time_to_expiration.append(T)

    in_the_money = np.array(in_the_money).reshape(-1)
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(in_the_money, time_to_expiration, implied_volatilities, cmap='viridis', linewidth=0.1, antialiased=True, alpha=0.8)


    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)       
    cbar.set_label('Implied Volatility', rotation=270, labelpad=15)


    ax.set_xlabel('In the Money Ratio')
    ax.set_ylabel('Time to Expiration (Years)')
    ax.set_zlabel('Implied Volatility')


    plt.title('Implied Volatility Surface for Call Options')

    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    plt.show()

  
    
    
