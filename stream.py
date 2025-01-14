import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from decimal import Decimal, getcontext

# Function to compute default probability using Merton's model
def default_probability(V, B, r, sigma, T, t, precision=20):
    """
    Computes the real-world probability of default according to Merton's model.
    """
    getcontext().prec = precision

    if B != 0:
        V_d = Decimal(float(V))
        B_d = Decimal(float(B))
        r_d = Decimal(float(r))
        sigma_d = Decimal(float(sigma))
        T_d = Decimal(T)
        arg = (np.log(float(B_d / V_d)) - float(Decimal('0.5') * sigma_d**2)) / float(sigma_d * np.sqrt(T_d - t))
        return float(norm.cdf(arg))
    else:
        return 0


# Streamlit App
st.title("Interactive Merton Model: Impact of Volatility on Probability of Default (PD)")
st.write("""
This interactive app shows how changing the **volatility** (σ) of the firm's asset impacts the **Probability of Default (PD)** 
based on Merton's model. 
Use the sliders and inputs to adjust the parameters.
""")



# Sidebar inputs for model parameters
st.sidebar.header("Model Parameters")
V = st.sidebar.number_input("Firm's Asset Value (V)", min_value=1.0, value=100.0, step=1.0)
B = st.sidebar.number_input("Debt Due at Maturity (B)", min_value=1.0, value=80.0, step=1.0)
r = st.sidebar.slider("Risk-free Interest Rate (r)", 0.0, 0.1, 0.03, 0.005)
T = st.sidebar.slider("Time to Maturity (T in years)", 0.1, 10.0, 1.0, 0.1)
t = 0  

with st.sidebar:
    for _ in range(3):
        st.write('')
    st.markdown("""*Made by: <br> Giada Martini, Francesco Mazzolin, Francesco Salvagnin, Nicolas Stecca*""", unsafe_allow_html=True)

# Volatility Slider for Interaction
sigma_selected = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)

# Compute probability of default with selected volatility
pd = default_probability(V, B, r, sigma_selected, T, t)
st.subheader(f"Probability of Default (PD) for σ = {sigma_selected:.2f}: {f'{pd:.3%}'}")

# Plot PD vs Volatility
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
