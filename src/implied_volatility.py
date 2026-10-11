import numpy as np

from black_scholes import black_scholes_price, option_vega


def no_arbitrage_bounds(S, K, T, r, option_type):
    # a European option price must sit between these two values
    discounted_K = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - discounted_K, 0.0), S
    elif option_type == "put":
        return max(discounted_K - S, 0.0), discounted_K
    else:
        raise ValueError("This is not a valid option type")


def implied_volatility_bisection(market_price, S, K, T, r, option_type,
                                 low=1e-6, high=5.0, tol=1e-8, max_iter=200):
    f_low = black_scholes_price(S, K, T, r, low, option_type) - market_price
    f_high = black_scholes_price(S, K, T, r, high, option_type) - market_price
    if f_low * f_high > 0:
        return np.nan                      # no volatility in [low, high] matches this price

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = black_scholes_price(S, K, T, r, mid, option_type) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)


def implied_volatility(market_price, S, K, T, r, option_type,
                       sigma0=0.20, tol=1e-8, max_iter=100):
    lower, upper = no_arbitrage_bounds(S, K, T, r, option_type)
    # prices outside the bounds (or with zero time value) carry no usable volatility information
    if market_price <= lower + 1e-8 or market_price >= upper:
        return np.nan

    sigma = sigma0
    for _ in range(max_iter):
        diff = black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        if abs(diff) < tol:
            return sigma
        vega = option_vega(S, K, T, r, sigma)
        if vega < 1e-10:                   # Newton step would explode, so switch method
            break
        sigma_new = sigma - diff / vega
        if sigma_new <= 0 or sigma_new > 5:
            break
        sigma = sigma_new

    return implied_volatility_bisection(market_price, S, K, T, r, option_type)


if __name__ == "__main__":
    S, K, T, r = 100, 105, 0.5, 0.05
    for true_sigma in [0.10, 0.25, 0.60]:
        for option_type in ["call", "put"]:
            price = black_scholes_price(S, K, T, r, true_sigma, option_type)
            print(option_type, true_sigma, implied_volatility(price, S, K, T, r, option_type))