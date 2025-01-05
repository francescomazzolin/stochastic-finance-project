#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
'''
 The jumps are composed by two elements: Arrival and Size.
 
 Starting from the size it is measured in log-Normal as BS. The change in price due to the 
 jumps is dSt = StYt - St.   Where Yt is the size of the jump.
 So for example If I have a jump of size 1.25. --> dSt = St*1.25 - St = St(1.25 - 1) (percentage change)
 So the formula to show the jump size is: dSt/St = Yt - 1

 The Arrival is based on a Poisson Distribution: dNt = {1,  lambda*dt
                                                        0,  1 - lambda*dt}

So the Overall Jump Process is done by adding to the Usual BS model the component (Yt - 1)*dNt

But, the Process also add a drift                                                 
''' 

def merton_jumps_plot(V0, sigma, r, T, M, N, lam, m, v, K):
    """
    Generate Monte Carlo simulated asset paths using Merton's Jump-Diffusion Model.

    Parameters:
    - V0 (float): Initial asset value.
    - sigma (float): Asset volatility.
    - r (float): Risk-free rate.
    - T (float): Time horizon (years).
    - M (int): Number of paths.
    - N (int): Number of time steps.
    - lam (float): Jump intensity (average number of jumps per year).
    - m (float): Mean of the jump size distribution (lognormal mean).
    - v (float): Standard deviation of the jump size distribution.

    Returns:
    - np.ndarray: Simulated asset paths (shape: [M, N]).
    """
    dt = T / N  
    paths = np.zeros((M, N))  
    paths[:, 0] = V0  
    time = np.linspace(0, T, N)
    for t in range(1, N):
        
        z = np.random.standard_normal(M)
        jump_count = np.random.poisson(lam * dt, M)
        jump_sizes = np.sum(np.random.normal(m, v, (M, max(jump_count))) * (np.arange(max(jump_count)) < jump_count[:, None]), axis=1)
        
        # Geometric Brownian motion + jumps
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z + jump_sizes)

    # Plotting the paths
    plt.figure(figsize=(10, 6))
    for i in range(M):
        plt.plot(np.linspace(0, T, N), paths[i], lw=0.8)
    plt.axhline(y=K, color='r', linestyle='--', linewidth=2, label='Debt Threshold (D)')
    plt.fill_between(time,0, K, color='red', alpha=0.2, label='Default Region')
    plt.title("Merton Jump-Diffusion Model Simulated Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Asset Value")
    plt.yscale('log')
    plt.show()

    return None                                       
V0 = 3993446559000.0 # current stock price
K = 107525000000.0
T = 5 # time to maturity
r = 0 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam =1 # intensity of jump i.e. number of jumps per annum
N =252 * T # time steps
M = 20000 # number of paths to simulate
sigma = 0.225923 # annaul standard deviation , for weiner process

merton_jumps_plot(V0, sigma, r, T, M, N, lam, m, v, K)

# %%
def merton_jumps_default(V0, K, T, M, N, lam, m, v, r, sigma):
    """
    Monte Carlo simulation with Merton jump diffusion model.
    
    Parameters:
    - V0 (float): Initial asset value.
    - D (float): Debt threshold (default barrier).
    - T (float): Time horizon (years).
    - M (int): Number of Monte Carlo paths.
    - N (int): Number of time steps.
    - lam (float): Poisson intensity (mean number of jumps per unit time).
    - m (float): Mean jump size.
    - v (float): Volatility of the jump size.
    - r (float): Risk-free interest rate.
    - sigma (float): Asset volatility.
    
    Returns:
    - Equity_mean (float): Mean equity value at maturity.
    - Debt_mean (float): Mean debt value at maturity.
    - prob_default (float): Default probability as percentage.
    """
    
    dt = T / N  
    paths = np.zeros((M, N))  
    paths[:, 0] = V0  

    for t in range(1, N):
        z = np.random.standard_normal(M)  
        jump_counts = np.random.poisson(lam * dt, M)  
        jump_sizes = np.random.normal(m, v, M) 
        
        
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        ) * np.exp(jump_counts * jump_sizes)

    # Terminal values
    V_T = paths[:, -1]  
    S_T = np.maximum(V_T - K, 0)  
    B_T = V_T - S_T  

    
    defaulted_paths = np.any(paths < K, axis=1)
    print(f"Number of default paths: {np.sum(defaulted_paths)}")
    prob_default = (np.sum(defaulted_paths) / M) * 100  

    Equity_mean = np.mean(S_T)
    Debt_mean = np.mean(B_T)

    return Equity_mean, Debt_mean, prob_default

equity_mean, debt_mean, prob_default = merton_jumps_default(V0, K, T, M, N, lam, m, v, r, sigma)
print(f"Mean Equity Value: {equity_mean:.2f}")
print(f"Mean Debt Value: {debt_mean:.2f}")
print(f"Probability of Default: {prob_default:.2f}%")

