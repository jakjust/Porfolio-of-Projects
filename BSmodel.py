import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp 
import matplotlib.pyplot as plt
import seaborn as sns




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
    
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return delta_call, delta_put, gamma, theta_call, theta_put, vega




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

    st.subheader("Option Prices")
    st.write(f"**Call Option Price:** ${call_price:.2f}")
    st.write(f"**Put Option Price:** ${put_price:.2f}")

    
    delta_call, delta_put, gamma, theta_call, theta_put, vega = bsm_greeks(S, K, T, r, sigma)

    st.subheader("Option Greeks")
    st.markdown(f"- **Delta (Call):** {delta_call:.4f}")
    st.markdown(f"- **Delta (Put):** {delta_put:.4f}")
    st.markdown(f"- **Gamma:** {gamma:.4f}")
    st.markdown(f"- **Theta (Call):** {theta_call:.4f}")
    st.markdown(f"- **Theta (Put):** {theta_put:.4f}")
    st.markdown(f"- **Vega:** {vega:.4f}")

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