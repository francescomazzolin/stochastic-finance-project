import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from decimal import Decimal, getcontext
from analytical_functions import default_probability

st.title("Interactive Merton Model: Impact of Volatility on Probability of Default")
for _ in range(1):
    st.write("")
st.write("""
This plot represents how changing the **volatility** (σ) of the firm's asset impacts the **Probability of Default (PD)** 
based on Merton's model
""")


st.sidebar.header("Model Parameters")
V = st.sidebar.number_input("Firm's Asset Value (V)", min_value=1.0, value=100.0, step=1.0)
B = st.sidebar.number_input("Debt Due at Maturity (B)", min_value=1.0, value=80.0, step=1.0)
r = st.sidebar.slider("Risk-free Interest Rate (r)", 0.0, 0.1, 0.03, 0.005)
T = st.sidebar.slider("Time to Maturity (T in years)", 0.1, 5.0, 1.0, 0.1)
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
