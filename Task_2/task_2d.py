import numpy as np
from math_functions import nCr


def european_binomial_up_out(S_0, K, T, r, n, B, delta, sigma):
    # Length of a period
    h = T / n

    # Up and down factor
    u = np.exp((r - delta) * h + sigma * np.sqrt(h))
    d = np.exp((r - delta) * h - sigma * np.sqrt(h))

    # Probability of price moving up
    pu = (np.exp((r - delta) * h) - d) / (u - d)
    pd = 1 - pu

    # Calculating sum of end node values times probability for ending up in said node
    node_sum = 0
    for i in range(0, n + 1):
        S_T = S_0 * u ** i * d ** (n - i)
        value = nCr(n, i) * pu ** i * pd ** (n - i) * max(0, S_T - K)

        # If S_T is larger or equal to B it is knocked out and worthless
        if S_T >= B:
            p = 0
        # If not, we want the probability that it still has value
        else:
            p = 1 - np.exp((-2/(T * (sigma**2))) * abs(np.log(S_0/B) * np.log(S_T/B)))
        node_sum += value * p
    node_sum = np.exp(-r * h * n) * node_sum
    return node_sum


def task_2d_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    n = 500  # Number of binomial steps
    H = 150  # Barrier

    call_option_price = european_binomial_up_out(S_0, K, T, r, n, H, delta, sigma)

    print("Price of a european up-and-out call option with the given parameters: %f" % call_option_price)

    return call_option_price


