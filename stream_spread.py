import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from decimal import Decimal, getcontext
from analytical_functions import default_probability

def credit_spread_model(L, sigma, r, T, t):
    if K != 0:
        # 1) Riskless bond price (for face value K)
        p_0 = np.exp(-r * (T - t))

        # 2) Calculate d1 and d2
        d1 = (-(np.log((K*p_0)/V)) + (0.5*sigma**2) * (T - t)) / (sigma * np.sqrt(T -t))

        d2 = d1 - (sigma * np.sqrt(T - t))

        # 3) *Defaultable* bond price using Merton's debt formula
        #defaultable_bond = K * riskless * norm.cdf(d2) + V * (1 - norm.cdf(d1))

        # 4) Credit spread calculation: -1/(T - t) * ln(defaultable_bond / riskless_bond)
        credit_spread = -1/(T - t) * np.log(norm.cdf(d2) + (V/(K *p_0))*norm.cdf(-d1))

    else: 

        credit_spread = 0

    return credit_spread


st.title("Interactive Merton Model: Comparison with market credit spreads")
for _ in range(1):
    st.write("")
st.write("""
This plot compares the credit spread computed by the model for the given parameters vs the one observed in the market.
""")


st.sidebar.header("Model Parameters")
L = st.sidebar.number_input("Firm's Leverage Ratio (V)", min_value=0.0, value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Interest Rate (r)", min_value=-0.05, value=0.5, step=0.001)
#r = st.sidebar.slider("Risk-free Interest Rate (r)", 0.0, 0.1, 0.03, 0.005)
#T = st.sidebar.slider("Time to Maturity (T in years)", 0.1, 5.0, 1.0, 0.1)
t = 0  
with st.sidebar:
    st.divider()
    for _ in range(1):
        st.write('')
    st.markdown("""*Made by: <br> Giada Martini, Francesco Mazzolin, Francesco Salvagnin, Nicolas Stecca*""", unsafe_allow_html=True)

# Volatility Slider for Interaction
st.markdown("<h5 style='font-size:22px;'>Volatility (σ)</h5>", unsafe_allow_html=True)
sigma_selected = st.slider("", 0.01, 1.0, 0.2, 0.1)  
pd = default_probability(V, B, r, sigma_selected, T, t)
st.subheader(f"Probability of Default (PD) : {f'{pd:.3%}'}")    


volatility_range = np.linspace(0.01, 1.0, 100)
pd_values = [default_probability(V, B, r, sigma, T, t) for sigma in volatility_range]

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(volatility_range, pd_values, label='PD vs Volatility (σ)', linewidth=2)
ax.axvline(sigma_selected, color='red', linestyle='--', label=f'Current σ = {sigma_selected:.2f}')
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Probability of Default (PD)")
ax.set_title("Impact of Volatility on Probability of Default (Merton Model)")
st.pyplot(fig)
