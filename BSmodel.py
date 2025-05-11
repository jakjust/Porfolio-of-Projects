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


def bsm_greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta_call = sp.diff(bsm_call(S, K, T, r, sigma), S).evalf()
    delta_put = sp.diff(bsm_put(S, K, T, r, sigma), S).evalf()
    gamma = sp.diff(delta_call, S).evalf()
    theta_call = sp.diff(bsm_call(S, K, T, r, sigma), T).evalf()
    theta_put = sp.diff(bsm_put(S, K, T, r, sigma), T).evalf()
    vega = sp.diff(bsm_call(S, K, T, r, sigma), sigma).evalf()
    rho_call = sp.diff(bsm_call(S, K, T, r, sigma), r).evalf()
    rho_put = sp.diff(bsm_put(S, K, T, r, sigma), r).evalf()
    
    return delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put 


ticker = yf.Ticker("AAPL")
expirations = ticker.options
opt_date = expirations[2]   
opt_chain = ticker.option_chain(opt_date)   
calls = opt_chain.calls
calls = calls[(calls["impliedVolatility"] > 0.01)]
puts = opt_chain.puts   
puts = puts[(puts["impliedVolatility"] > 0.01)] 


def expiration_date(contractSymbol):
    date = contractSymbol[4:10]
    year = '20' + date[0:2]
    month = date[2:4]
    day = date[4:6]
    return f"{year}-{month}-{day}"

calls["expirations"] = calls["contractSymbol"].apply(expiration_date)
puts["expirations"] = puts["contractSymbol"].apply(expiration_date)

calls["timeToExpiration"] = (pd.to_datetime(calls["expirations"]) - pd.to_datetime("today")).dt.days /365
puts["timeToExpiration"] = (pd.to_datetime(puts["expirations"]) - pd.to_datetime("today")).dt.days /365







def loss_call(S, K, T, r, sigma_guess, price):
    theorietical_price = bsm_call(S, K, T, r, sigma_guess) 
    
    actual_price = price
    
    
    return theorietical_price - actual_price



def loss_put(S, K, T, r, sigma_guess, price):
    theorietical_price = bsm_put(S, K, T, r, sigma_guess) 
    
    actual_price = price
    
    
    return theorietical_price - actual_price






loss_grad = grad(loss_call, argnums=4)

def solve_iv_call(S, K, T, r, price, sigma_guess=0.2, 
             N_iter = 20, epsilon = 0.001, verbose = True):
    simga = sigma_guess
    for i in range(N_iter):
        loss_val = loss_call(S, K, T, r, sigma, price)
        
        if abs(loss_val) < epsilon:
            break
        else:
            loss_grad_val = loss_grad(S, K, T, r, sigma, price)
            sigma = sigma - loss_val/loss_grad_val
    return sigma

loss_grad = grad(loss_put, argnums=4)

def solve_iv_put(S, K, T, r, price, sigma_guess=0.2,
             N_iter = 20, epsilon = 0.001, verbose = True):
    simga = sigma_guess
    for i in range(N_iter):
        loss_val = loss_put(S, K, T, r, sigma, price)
        
        if abs(loss_val) < epsilon:
            break
        else:
            loss_grad_val = loss_grad(S, K, T, r, sigma, price)
            sigma = sigma - loss_val/loss_grad_val
    return sigma




            
        


    

st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")
st.title("Black-Scholes Option Pricing Model")



S = float(st.number_input("Stock Price (S)", min_value=0.0, value=100.0))
K = float(st.number_input("Strike Price (K)", min_value=0.0, value=100.0))
T = float(st.number_input("Time to Expiration (T in years)", min_value=0.0, value=1.0))
r = float(st.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, format="%.4f"))
sigma = float(st.number_input("Volatility (σ)", min_value=0.0, value=0.2, format="%.4f"))


if st.button("Calculate Option Prices"):
   
    call_price = bsm_call(S, K, T, r, sigma)
    put_price = bsm_put(S, K, T, r, sigma)
    impliedvolatilitycall = solve_iv_call(S, K, T, r,)
    impliedvolatilityput = solve_iv_put(S, K, T, r,)

    st.subheader("Option Prices")
    st.write(f"**Call Option Price:** ${call_price:.2f}")
    st.write(f"**Put Option Price:** ${put_price:.2f}")
    
    st.subheader("Implied Volatility")
    st.write(f"**Implied Volatility (Call):** {impliedvolatilitycall:.4f}")
    st.write(f"**Implied Volatility (Put):** {impliedvolatilityput:.4f}")

    
    delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put = bsm_greeks(S, K, T, r, sigma)

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