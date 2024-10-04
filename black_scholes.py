import math
from scipy.stats import norm

def black_scholes_option_pricing(S: float, K: float, T: float, r: float, sigma: float, type: str) -> float:
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(S, K, T, r, type, market_price, tol=1e-8, max_iterations=100):
    """
    To calculate implied volatility using the Newton-Raphson method
    """
    sigma = 0.2  # initial guess for implied volatility
    for i in range(max_iterations):
        price = black_scholes_option_pricing(S, K, T, r, sigma, type)
        vega = (S * norm.pdf((math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))) * math.sqrt(T))
        price_diff = market_price - price  # difference between market price and model price
        
        if abs(price_diff) < tol:
            return sigma
        
        sigma = sigma + price_diff / vega  # Newton-Raphson step
    
    return None