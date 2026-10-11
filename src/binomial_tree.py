import numpy as np

def crr_parameters(T, r, sigma, N):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    return dt,u,d,p

def stock_price_tree(S, u, d, N):
    tree = []
    for i in range (N+1): # need N+1 since 0 indexed would give us one too few
        row = []
        for j in range (i+1): # same i+1 as above
            row.append(S*u**j*d**(i-j))
        tree.append(row)
    terminal_stock_prices = tree[-1]
    return tree, terminal_stock_prices

def terminal_option_value(terminal_stock_prices, K, option_type):
    if option_type == "call":
        payoffs = []
        for St in terminal_stock_prices:
            payoffs.append(max(St-K,0)) 
        return payoffs
    elif option_type == "put":
        payoffs = []
        for St in terminal_stock_prices:
            payoffs.append(max(K-St,0))
        return payoffs
    else:
        raise ValueError("This is not a valid option type")
    # righthand member of each pair is the up outcome, and the lefthand member is the down outcome

def backward_option_value(payoffs, p, r, dt):
    backward_calculated_option_value = [] #  this function calculates the pre
    for i in range (len(payoffs)-1):
        value = np.exp(-r*dt)*(p*payoffs[i+1]+(1-p)*payoffs[i]) # this needs to be the order of 
        backward_calculated_option_value.append(value)    #payoffs [i] and i+1 since Vu is the righthand element ie i+1
    return backward_calculated_option_value
        
def binomial_option_price(S, K, T, r, sigma, N, option_type):
    dt,u,d,p = crr_parameters(T, r, sigma, N)
    tree, terminal_stock_prices = stock_price_tree(S,u,d,N)
    payoffs = terminal_option_value(terminal_stock_prices, K, option_type)
    for i in range (N):
        payoffs = backward_option_value(payoffs,p,r,dt)
    return payoffs[0]

def american_backward_option_value(option_values, stock_prices, p, r, dt, K, option_type):
    new_option_values = []
    for i in range(len(stock_prices)): # looping through the list of stock_prices
        continuation_value = np.exp(-r*dt) * (p*option_values[i+1] + (1-p)*option_values[i])

        if option_type == "call":
            exercise_value = max(stock_prices[i] - K, 0)
        elif option_type == "put":
            exercise_value = max(K - stock_prices[i], 0)
        else:
            raise ValueError("This is not a valid option type")
        new_option_values.append(max(continuation_value, exercise_value))

    return new_option_values

def american_binomial_option_price(S, K, T, r, sigma, N, option_type):
    dt, u, d, p = crr_parameters(T, r, sigma, N)
    tree, terminal_stock_prices = stock_price_tree(S, u, d, N)
    option_values = terminal_option_value(terminal_stock_prices, K, option_type)
    for i in range(N):
        stock_prices = tree[N - 1 - i] # working backwards through the tree, stock price at a particular step
        option_values = american_backward_option_value(option_values,stock_prices,p,r,dt,K,option_type)
    return option_values[0] # theoretical value of american option today

if __name__ == "__main__":
    print(binomial_option_price(100, 100, 1, 0.05, 0.20, 1000, "call"))
    print(american_binomial_option_price(100, 100, 1, 0.05, 0.20, 1000, "call"))

    print(binomial_option_price(100, 100, 1, 0.05, 0.20, 1000, "put"))
    print(american_binomial_option_price(100, 100, 1, 0.05, 0.20, 1000, "put"))

    for N in [10, 50, 100, 500, 1000]:
        price = american_binomial_option_price(
            100, 100, 1, 0.05, 0.20, N, "put"
        )
        print(N, price)