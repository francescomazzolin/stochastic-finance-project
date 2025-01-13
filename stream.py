import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Title and Description
st.title("Merton Model: Interactive Probability of Default")
st.write("""
This interactive app demonstrates how **volatility (σ)** impacts the **Probability of Default (PD)** of a firm
based on the Merton model.
""")

# Sidebar Inputs
st.sidebar.header("Model Parameters")
asset_value = st.sidebar.number_input("Firm's Asset Value (V_A)", min_value=1.0, value=100.0, step=1.0)
debt = st.sidebar.number_input("Debt Due at Maturity (D)", min_value=1.0, value=80.0, step=1.0)
risk_free_rate = st.sidebar.slider("Risk-free Interest Rate (r)", 0.0, 0.1, 0.03, 0.005)
time_to_maturity = st.sidebar.slider("Time to Maturity (T in years)", 0.1, 10.0, 1.0, 0.1)
volatility = st.sidebar.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)

# Merton Model Calculations
d1 = (np.log(asset_value / debt) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
d2 = d1 - volatility * np.sqrt(time_to_maturity)
pd = norm.cdf(-d2)

# Display Probability of Default
st.subheader(f"Calculated Probability of Default (PD): {pd:.4f}")

# Plotting the PD against different volatilities
volatility_range = np.linspace(0.01, 1.0, 100)
pd_values = [norm.cdf(-(np.log(asset_value / debt) + (risk_free_rate + 0.5 * v ** 2) * time_to_maturity) / (v * np.sqrt(time_to_maturity)) - v * np.sqrt(time_to_maturity)) for v in volatility_range]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(volatility_range, pd_values, label='PD vs Volatility (σ)')
ax.axvline(volatility, color='red', linestyle='--', label=f'Current σ = {volatility:.2f}')
ax.set_xlabel("Volatility (σ)")
ax.set_ylabel("Probability of Default (PD)")
ax.set_title("Impact of Volatility on Probability of Default")
ax.legend()
ax.grid(True)

# Show plot
st.pyplot(fig)
