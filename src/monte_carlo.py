import numpy as np


def GBM_terminal_stock_prices(S, T, r, sigma, n):
    Z = np.random.standard_normal(n)
    terminal_stock_prices = S * np.exp((r - 0.5 * sigma**2) * T+ sigma * np.sqrt(T) * Z)
    return terminal_stock_prices
# outputs a numpy array of randomly generated stock prices based on GBM 

def option_payoffs(terminal_stock_prices, K, option_type):
    if option_type == "call":
        payoffs = np.maximum(terminal_stock_prices-K,0)
        return payoffs
    elif option_type == "put":
        payoffs = np.maximum  (K-terminal_stock_prices, 0)
        return payoffs
    else:
        raise ValueError("This is not a valid option type")

def monte_carlo_price(payoffs, T, r):
    present_day_value = np.exp(-r*T)*np.mean(payoffs)
    return present_day_value

def monte_carlo_standard_error(payoffs, T, r):
    N = len(payoffs)
    discounted_payoffs = payoffs*np.exp(-r*T)
    average_discounted_payoffs = np.mean(discounted_payoffs)
    transformed_payoffs = (discounted_payoffs - average_discounted_payoffs)**2
    standard_deviation = np.sqrt((1/(N-1))*np.sum(transformed_payoffs))
    se = standard_deviation/np.sqrt(N)
    return se

def monte_carlo_option_price(S, K, T, r, sigma, n, option_type):
    terminal_stock_prices = GBM_terminal_stock_prices(S, T, r, sigma, n)
    payoffs = option_payoffs(terminal_stock_prices, K, option_type)
    return monte_carlo_price(payoffs, T, r), monte_carlo_standard_error(payoffs, T, r)

def GBM_antithetic_terminal_stock_prices(S, T, r, sigma, n):
    Z = np.random.standard_normal(int(n/2))
    terminal_stock_prices = S * np.exp((r - 0.5 * sigma**2) * T+ sigma * np.sqrt(T) * Z)
    antithetic_terminal_stock_prices = S * np.exp((r - 0.5 * sigma**2) * T- sigma * np.sqrt(T) * Z)
    return terminal_stock_prices, antithetic_terminal_stock_prices
# outputs a numpy array of randomly generated stock prices based on GBM and the antithetic pair

def antithetic_option_payoffs(terminal_stock_prices,antithetic_terminal_stock_prices, K, option_type):
    if option_type == "call":
        payoffs = np.maximum(terminal_stock_prices-K,0)
        antithetic_payoffs = np.maximum(antithetic_terminal_stock_prices-K,0)
        
    elif option_type == "put":
        payoffs = np.maximum  (K-terminal_stock_prices, 0)
        antithetic_payoffs = np.maximum(K-antithetic_terminal_stock_prices,0)
        
    else:
        raise ValueError("This is not a valid option type")
    combined_average_payoff = (payoffs + antithetic_payoffs)/2
    return combined_average_payoff

def antithetic_monte_carlo_option_price(S, K, T, r, sigma, n, option_type):
    terminal_stock_prices, antithetic_terminal_stock_prices = GBM_antithetic_terminal_stock_prices(S, T, r, sigma, n)
    payoffs = antithetic_option_payoffs(terminal_stock_prices,antithetic_terminal_stock_prices, K, option_type)
    return monte_carlo_price(payoffs, T, r), monte_carlo_standard_error(payoffs, T, r)

# NOW DOING MONTE CARLO GREEKS CALCULATIONS

def GBM_terminal_stock_prices_from_Z(S, T, r, sigma, Z):
    terminal_stock_prices = S * np.exp( (r - 0.5 * sigma**2) * T+ sigma * np.sqrt(T) * Z)
    return terminal_stock_prices

def monte_carlo_delta(S, K, T, r, sigma, n, option_type, h): 
    Z = np.random.standard_normal(n) 
    terminal_prices_plus = GBM_terminal_stock_prices_from_Z(S+h, T, r, sigma, Z) # passing in S+-h 
    terminal_prices_minus = GBM_terminal_stock_prices_from_Z(S-h, T, r, sigma, Z) #as the stock price
    
    payoffs_plus = option_payoffs(terminal_prices_plus, K, option_type)
    payoffs_minus = option_payoffs(terminal_prices_minus, K, option_type)
    
    price_plus = monte_carlo_price(payoffs_plus, T, r)
    price_minus = monte_carlo_price(payoffs_minus, T, r)
    
    delta = (price_plus - price_minus) / (2*h)
    
    return delta

def monte_carlo_gamma(S, K, T, r, sigma, n, option_type, h=1.0):
    Z = np.random.standard_normal(n) 
    terminal_prices_plus = GBM_terminal_stock_prices_from_Z(S+h, T, r, sigma, Z) # passing in S+-h 
    terminal_prices_minus = GBM_terminal_stock_prices_from_Z(S-h, T, r, sigma, Z) #as the stock price
    terminal_prices_no_h = GBM_terminal_stock_prices_from_Z(S, T, r, sigma, Z)

    payoffs_plus = option_payoffs(terminal_prices_plus, K, option_type)
    payoffs_minus = option_payoffs(terminal_prices_minus, K, option_type)
    payoffs_no_h = option_payoffs(terminal_prices_no_h, K, option_type)
        

    price_plus = monte_carlo_price(payoffs_plus, T, r)
    price_minus = monte_carlo_price(payoffs_minus, T, r)
    price_no_h = monte_carlo_price(payoffs_no_h, T, r)

    gamma = (price_plus - 2*price_no_h + price_minus)/h**2
    return gamma

def monte_carlo_vega(S, K, T, r, sigma, n, option_type, h): 
    Z = np.random.standard_normal(n) 
    terminal_prices_plus = GBM_terminal_stock_prices_from_Z(S, T, r, sigma+h, Z) # sigma +-h 
    terminal_prices_minus = GBM_terminal_stock_prices_from_Z(S, T, r, sigma-h, Z) #as volatility
    
    payoffs_plus = option_payoffs(terminal_prices_plus, K, option_type)
    payoffs_minus = option_payoffs(terminal_prices_minus, K, option_type)
    
    price_plus = monte_carlo_price(payoffs_plus, T, r)
    price_minus = monte_carlo_price(payoffs_minus, T, r)
    
    vega = (price_plus - price_minus) / (2*h)
    
    return vega

def monte_carlo_rho(S, K, T, r, sigma, n, option_type, h): 
    Z = np.random.standard_normal(n) 
    terminal_prices_plus = GBM_terminal_stock_prices_from_Z(S, T, r+h, sigma, Z) # sigma +-h 
    terminal_prices_minus = GBM_terminal_stock_prices_from_Z(S, T, r-h, sigma, Z) #as volatility
    
    payoffs_plus = option_payoffs(terminal_prices_plus, K, option_type)
    payoffs_minus = option_payoffs(terminal_prices_minus, K, option_type)
    
    price_plus = monte_carlo_price(payoffs_plus, T, r+h)
    price_minus = monte_carlo_price(payoffs_minus, T, r-h)
    
    rho = (price_plus - price_minus) / (2*h)
    
    return rho

def monte_carlo_theta(S, K, T, r, sigma, n, option_type, h): 
    Z = np.random.standard_normal(n) 
    terminal_prices_plus = GBM_terminal_stock_prices_from_Z(S, T+h, r, sigma, Z) # T +-h 
    terminal_prices_minus = GBM_terminal_stock_prices_from_Z(S, T-h, r, sigma, Z) #as time
    
    payoffs_plus = option_payoffs(terminal_prices_plus, K, option_type)
    payoffs_minus = option_payoffs(terminal_prices_minus, K, option_type)
    
    price_plus = monte_carlo_price(payoffs_plus, T+h, r)
    price_minus = monte_carlo_price(payoffs_minus, T-h, r)
    
    theta = -(price_plus - price_minus) / (2*h)
    
    return theta

if __name__ == "__main__":
    price, se = monte_carlo_option_price(
        100, 100, 1, 0.05, 0.20, 10000, "call"
    )

    print(price)
    print(se)

if __name__ == "__main__":
    price, se = antithetic_monte_carlo_option_price(
        100, 100, 1, 0.05, 0.20, 10000, "call"
    )

    print(price)
    print(se)

# THIS IS NOW FOR LONGSTAFF SCHWARTZ FOR AMERICAN OPTIONS

def GBM_paths(S, T, r, sigma, n_paths, n_steps, antithetic=False):
    dt = T / n_steps
    if antithetic:
        half = n_paths // 2
        Z_half = np.random.standard_normal((half, n_steps))
        Z = np.vstack((Z_half, -Z_half))   # first half, then its mirror image
    else:
        Z = np.random.standard_normal((n_paths, n_steps))

    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((Z.shape[0], n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(log_paths)
    return paths