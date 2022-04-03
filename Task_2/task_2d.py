import numpy as np
from math_functions import nCr


def european_binomial_up_out(S_0, k, T, r, n, B, delta, sigma):
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

        # Value of stock at time t
        S_T = S_0 * np.power(u, n - i) * np.power(d, i)

        # Intrinsic value of call at i down moves
        int_val = np.max([S_T - k, 0])

        val = int_val * prob

        # If the ending value of the stock, s_t,
        # is above the barrier the intrinsic value of the knock-out option is zero
        if S_T >= B:
            p = 0

        # If the ending value of the stock, s_t, is below the barrier H
        # we have to correct for the probability of hitting it earlier
        else:
            # Probability of hitting the barrier before ending up at final value s_t < H
            prob_barrier = np.exp(- 2 / (np.power(sigma, 2) * T) * abs(np.log(S_0 / B) * np.log(S_T / B)))

            # Probability of not hitting the barrier, which would mean that the option stays alive and has value
            p = 1 - prob_barrier

        # Summing up after multiplying probability and intrinsic value
        node_sum += val * p

    # Discount the sum
    option_price = np.exp(- r * T) * node_sum

    return option_price


def task_2d_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    n = 3  # Number of binomial steps
    H = 150  # Barrier

    call_option_price = european_binomial_up_out(S_0, K, T, r, n, H, delta, sigma)

    print("Price of a european up-and-out call option with the given parameters: %f" % call_option_price)

    return call_option_price


