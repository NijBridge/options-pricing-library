from src import black_scholes as bs, binomial_tree as bt, monte_carlo as mc
import math
import numpy as np

"""
S = 100
K = 100 Use these values as benchmark for all the tests
T = 1
r = 0.05
sigma = 0.20
"""

def test_call_price():
    price = bs.black_scholes_price(100, 100, 1, 0.05, 0.20, "call")
    assert math.isclose(price, 10.4506, rel_tol=1e-4)

def test_put_price():
    price = bs.black_scholes_price(100, 100, 1, 0.05, 0.20, "put")
    assert math.isclose(price, 5.5735, rel_tol=1e-4)

def test_call_delta():
    delta = bs.option_delta(100, 100, 1, 0.05, 0.20, "call")
    assert math.isclose(delta, 0.6368306512, rel_tol=1e-4)


def test_put_delta():
    delta = bs.option_delta(100, 100, 1, 0.05, 0.20, "put")
    assert math.isclose(delta, -0.3631693488, rel_tol=1e-4)


def test_gamma():
    gamma = bs.option_gamma(100, 100, 1, 0.05, 0.20)
    assert math.isclose(gamma, 0.01876201735, rel_tol=1e-4)


def test_vega():
    vega = bs.option_vega(100, 100, 1, 0.05, 0.20)
    assert math.isclose(vega, 37.5240346917, rel_tol=1e-4)


def test_call_theta():
    theta = bs.option_theta(100, 100, 1, 0.05, 0.20, "call")
    assert math.isclose(theta, -6.4140275464, rel_tol=1e-4)


def test_put_theta():
    theta = bs.option_theta(100, 100, 1, 0.05, 0.20, "put")
    assert math.isclose(theta, -1.6578804239, rel_tol=1e-4)


def test_call_rho():
    rho = bs.option_rho(100, 100, 1, 0.05, 0.20, "call")
    assert math.isclose(rho, 53.2324815454, rel_tol=1e-4)


def test_put_rho():
    rho = bs.option_rho(100, 100, 1, 0.05, 0.20, "put")
    assert math.isclose(rho, -41.8904609387, rel_tol=1e-4)

# ---------- CRR PARAMETERS ----------

def test_crr_parameters():
    dt, u, d, p = bt.crr_parameters(1, 0.05, 0.20, 100)

    assert math.isclose(dt, 0.01, rel_tol=1e-9)
    assert math.isclose(u * d, 1.0, rel_tol=1e-9)
    assert 0 < p < 1


# ---------- STOCK PRICE TREE ----------

def test_stock_price_tree():
    tree, terminal_prices = bt.stock_price_tree(100, 1.1, 1/1.1, 2)

    assert math.isclose(tree[0][0], 100)
    assert math.isclose(tree[1][0], 100 / 1.1)
    assert math.isclose(tree[1][1], 110)
    assert math.isclose(tree[2][0], 100 / (1.1**2))
    assert math.isclose(tree[2][1], 100)
    assert math.isclose(tree[2][2], 100 * (1.1**2))

    assert terminal_prices == tree[-1]


# ---------- TERMINAL OPTION VALUES ----------

def test_terminal_call_payoffs():
    prices = [80, 100, 120]

    payoffs = bt.terminal_option_value(prices, 100, "call")

    assert payoffs == [0, 0, 20]


def test_terminal_put_payoffs():
    prices = [80, 100, 120]

    payoffs = bt.terminal_option_value(prices, 100, "put")

    assert payoffs == [20, 0, 0]


def test_invalid_option_type():
    prices = [80, 100, 120]

    try:
        bt.terminal_option_value(prices, 100, "banana")
        assert False
    except ValueError:
        pass


# ---------- EUROPEAN BACKWARD INDUCTION ----------

def test_backward_option_value():
    payoffs = [0, 10, 20]

    result = bt.backward_option_value(
        payoffs,
        p=0.5,
        r=0,
        dt=1
    )

    assert result == [5, 15]


# ---------- EUROPEAN BINOMIAL PRICE ----------

def test_binomial_call_converges_to_black_scholes():
    price = bt.binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "call"
    )

    assert math.isclose(price, 10.4506, abs_tol=0.01)


def test_binomial_put_converges_to_black_scholes():
    price = bt.binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "put"
    )

    assert math.isclose(price, 5.5735, abs_tol=0.01)


# ---------- AMERICAN BACKWARD INDUCTION ----------

def test_american_backward_option_value():
    option_values = [0, 0, 20]
    stock_prices = [90, 110]

    result = bt.american_backward_option_value(
        option_values,
        stock_prices,
        p=0.5,
        r=0,
        dt=1,
        K=100,
        option_type="call"
    )

    assert result == [0, 10]


# ---------- AMERICAN OPTIONS ----------

def test_american_call_equals_european_call():
    european = bt.binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "call"
    )

    american = bt.american_binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "call"
    )

    assert math.isclose(american, european, abs_tol=1e-10)


def test_american_put_greater_than_european_put():
    european = bt.binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "put"
    )

    american = bt.american_binomial_option_price(
        100, 100, 1, 0.05, 0.20, 1000, "put"
    )

    assert american > european


def test_american_put_converges():
    prices = []

    for N in [100, 500, 1000]:
        price = bt.american_binomial_option_price(
            100, 100, 1, 0.05, 0.20, N, "put"
        )
        prices.append(price)

    assert abs(prices[2] - prices[1]) < abs(prices[1] - prices[0])


def test_american_option_has_at_least_intrinsic_value():
    call = bt.american_binomial_option_price(
        120, 100, 1, 0.05, 0.20, 500, "call"
    )

    put = bt.american_binomial_option_price(
        80, 100, 1, 0.05, 0.20, 500, "put"
    )

    assert call >= 20
    assert put >= 20

def test_simulate_terminal_stock_prices():
    prices = mc.GBM_terminal_stock_prices(100, 1, 0.05, 0.20, 1000)

    assert len(prices) == 1000
    assert np.all(prices > 0)

def test_option_payoffs():
    terminal_prices = np.array([80, 90, 100, 110, 120])

    call_payoffs = mc.option_payoffs(terminal_prices, 100, "call")
    put_payoffs = mc.option_payoffs(terminal_prices, 100, "put")

    assert np.array_equal(call_payoffs, np.array([0, 0, 0, 10, 20]))
    assert np.array_equal(put_payoffs, np.array([20, 10, 0, 0, 0]))

def test_monte_carlo_price():
    payoffs = np.array([0, 10, 20, 30])

    price = mc.monte_carlo_price(payoffs, 1, 0.05)

    expected = np.exp(-0.05) * 15

    assert np.isclose(price, expected)

#def test_monte_carlo_option_price():
    #price = mc.monte_carlo_option_price(
     #   100, 100, 1, 0.05, 0.20, 100000, "call")
    #black_scholes_price = 10.4506
    #assert np.isclose(price, black_scholes_price, atol=0.1)
# not sure how to test this yet, need to find a way

def test_monte_carlo_standard_error():
    payoffs = np.array([10, 20, 30, 40])

    se = mc.monte_carlo_standard_error(payoffs, 1, 0)

    assert np.isclose(se, 6.454972)
    
