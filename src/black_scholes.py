import math
from scipy.stats import norm
import numpy as np

"""
d1 roughly measures how far in-the-money the option is, in units 
of "standard deviations of movement," adjusted for drift.

d2 is the same thing but for the probability the option 
actually gets exercised.

following function defines these parameters
"""

"""
S - current underlying price r - risk free interest rate
K - strike price T - time to expiry in years
sigma - volatility of underlying
"""
def d1_d2(S, K, T, r, sigma):
    d1  = ((np.log(S/K))+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -  sigma*np.sqrt(T) # this function calculates d1 and d2
    return d1, d2

def black_scholes_price(S, K, T, r, sigma, option_type):
    d1, d2 = d1_d2(S, K, T, r, sigma) # caluclate the standardised values again
    
    if option_type == "call":
        C = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
        return C 
    # this function calcualtes call and put option prices    
    elif option_type == "put":
        P = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return P       
    else:
        raise ValueError("This is not a valid option type")

# GREEKS CALCULATION

def option_delta(S, K, T, r, sigma, option_type):
    d1, d2 = d1_d2(S, K, T, r, sigma) # call this function to calculate d1

    if option_type == "call":
        delta = norm.cdf(d1)
        return delta
    # calculates the option delta given if its a put or a call
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        return delta
    else:
        raise ValueError("This is not a valid option type")

def option_gamma(S,K,T,r,sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma) # call this function to calculate d1
    gamma = (norm.pdf(d1))/(S*sigma*np.sqrt(T))
    return gamma

def option_vega(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma) # call this function to calculate d1
    vega = S*norm.pdf(d1)*np.sqrt(T)
    return vega

def option_theta(S, K, T, r, sigma, option_type):
    d1, d2 = d1_d2(S, K, T, r, sigma) # call this function to calculate d1
    if option_type == "call":
        theta = ((-S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)))-r*K*np.exp(-r*T)*norm.cdf(d2)
        return theta
    # calculates the option theta given if its a put or a call
    elif option_type == "put":
        theta = ((-S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)))+r*K*np.exp(-r*T)*norm.cdf(-   d2)
        return theta
    else:
        raise ValueError("This is not a valid option type")

def option_rho(S, K, T, r, sigma, option_type):
    d1, d2 = d1_d2(S, K, T, r, sigma) # call this function to calculate d1
    if option_type == "call":
        rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        return rho
    # calculates the option rho given if its a put or a call
    elif option_type == "put":
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        return rho
    else:
        raise ValueError("This is not a valid option type")   

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.20

if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20

    print(black_scholes_price(S, K, T, r, sigma, "call"))
    print(option_delta(S, K, T, r, sigma, "call"))
    print(option_gamma(S, K, T, r, sigma))
    print(option_vega(S, K, T, r, sigma))
    print(option_theta(S, K, T, r, sigma, "call"))
    print(option_rho(S, K, T, r, sigma, "call"))