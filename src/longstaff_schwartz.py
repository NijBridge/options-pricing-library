import numpy as np
from numpy.polynomial.laguerre import lagvander

from monte_carlo import GBM_paths


def intrinsic_value(stock_prices, K, option_type):
    if option_type == "call":
        return np.maximum(stock_prices - K, 0)
    elif option_type == "put":
        return np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("This is not a valid option type")


def laguerre_basis(x, degree=3):
    # columns: exp(-x/2) * L0(x), exp(-x/2) * L1(x), ... up to L_degree(x)
    return np.exp(-x / 2)[:, None] * lagvander(x, degree)


def lsm_american_price(S, K, T, r, sigma, n_paths, n_steps, option_type,
                       degree=3, antithetic=True):
    dt = T / n_steps
    discount = np.exp(-r * dt)

    paths = GBM_paths(S, T, r, sigma, n_paths, n_steps, antithetic)

    # cashflows[i] = value (in time-t money) of path i's best plan so far
    cashflows = intrinsic_value(paths[:, -1], K, option_type)

    for t in range(n_steps - 1, 0, -1):
        cashflows = cashflows * discount           # bring value back one step to time t
        exercise = intrinsic_value(paths[:, t], K, option_type)
        itm = exercise > 0                         # only in-the-money paths get regressed

        if itm.sum() > degree + 1:                 # need enough points to fit the regression
            x = paths[itm, t] / K                  # normalise so the basis is well-scaled
            A = laguerre_basis(x, degree)
            coeffs, *_ = np.linalg.lstsq(A, cashflows[itm], rcond=None)
            continuation = A @ coeffs              # estimated continuation value

            exercise_now = exercise[itm] > continuation
            idx = np.where(itm)[0][exercise_now]   # paths where exercising beats continuing
            cashflows[idx] = exercise[idx]         # exercise: overwrite future cashflow

    # one last discount step from time 1 back to time 0
    if antithetic:
        half = len(cashflows) // 2
        samples = 0.5 * (cashflows[:half] + cashflows[half:])   # pair averages are independent
    else:
        samples = cashflows

    price = discount * np.mean(samples)
    se = discount * np.std(samples, ddof=1) / np.sqrt(len(samples))
    return price, se


if __name__ == "__main__":
    from binomial_tree import american_binomial_option_price
    from black_scholes import black_scholes_price

    S, K, T, r, sigma = 36, 40, 1, 0.06, 0.20
    price, se = lsm_american_price(S, K, T, r, sigma, 100000, 50, "put")
    print("LSM American put:     ", price, "+/-", se)
    print("Binomial American put:", american_binomial_option_price(S, K, T, r, sigma, 500, "put"))
    print("BS European put:      ", black_scholes_price(S, K, T, r, sigma, "put"))