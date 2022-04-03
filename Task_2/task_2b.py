import numpy as np
from math_functions import nCr
import matplotlib.pyplot as plt


def european_binomial(call, S_0, K, T, r, delta, sigma, n):
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
    for i in range(n + 1):

        # Probability for ending up at node with i down moves
        prob = np.power(pu, n - i) * np.power(pd, i) * nCr(n, i)

        # Value of stock at time T
        S_T = S_0 * np.power(u, n - i) * np.power(d, i)

        if call:
            # Intrinsic value of call at i down moves
            int_val = np.max([S_T - K, 0])

            # Summing up after multiplying probability and intrinsic value
            node_sum += int_val * prob

        else:
            # Intrinsic value of put at i down moves
            int_val = np.max([K - S_0 * np.power(u, n - i) * np.power(d, i), 0])

            # Summing up after multiplying probability and intrinsic value
            node_sum += int_val * prob

    # Discount the sum
    option_price = np.exp(- r * T) * node_sum

    return option_price


def binomial_steps_convergence():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility

    call_option_values = []
    put_option_values = []
    x_axis = []
    n_options = [1, 50, 100, 500]
    for n in n_options:
        call_option_values.append(european_binomial(True, S_0, K, T, r, delta, sigma, n))
        put_option_values.append(european_binomial(False, S_0, K, T, r, delta, sigma, n))
        x_axis.append(n)

    return call_option_values, put_option_values


def task_2b_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    n = 500  # Number of binomial steps

    euro_call = european_binomial(True, S_0, K, T, r, delta, sigma, n)
    euro_put = european_binomial(False, S_0, K, T, r, delta, sigma, n)

    print("Price of a european call with the given parameters: %f" % euro_call)
    print("Price of a european put with the given parameters: %f" % euro_put)