import numpy as np

from math_functions import nCr


def american_binomial(call, S_0, K, T, r, delta, sigma, n, i):
    # Length of a period
    h = T / n

    # Up and down factor
    u = np.exp((r - delta) * h + sigma * np.sqrt(h))
    d = np.exp((r - delta) * h - sigma * np.sqrt(h))

    # Probability of price moving up
    pu = (np.exp((r - delta) * h) - d) / (u - d)
    pd = 1 - pu

    if call:

        # If we are at max number of binomial steps, return edge node call option value
        if i == n:
            # Intrinsic value of call at current node
            int_val = np.max([S_0 - K, 0])

            return int_val

        # Value of the option at the "next" node when the price goes up
        recursive_value_up = american_binomial(True, S_0 * u, K, T, r, delta, sigma, n, i + 1)

        # Value of the option at the "next" node when the price goes down
        recursive_value_down = american_binomial(True, S_0 * d, K, T, r, delta, sigma, n, i + 1)

        # Return max of exercising the option early at current node or letting it go another binomial period
        return np.max([S_0 - K, np.exp(-r * h) * (recursive_value_up * pu + recursive_value_down * pd)])

    else:
        # If we are at max number of binomial steps, return edge node call option value
        if i == n:
            # Intrinsic value of put at current node
            int_val = np.max([K - S_0, 0])

            return int_val

        # Value of the option at the "next" node when the price goes up
        recursive_value_up = american_binomial(False, S_0 * u, K, T, r, delta, sigma, n, i + 1)

        # Value of the option at the "next" node when the price goes down
        recursive_value_down = american_binomial(False, S_0 * d, K, T, r, delta, sigma, n, i + 1)

        # Return max of exercising the option early at current node or letting it go another binomial period
        return np.max([K - S_0, np.exp(-r * h) * (recursive_value_up * pu + recursive_value_down * pd)])


def task_3a_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    n = 17  # Number of binomial steps

    american_call = american_binomial(True, S_0, K, T, r, delta, sigma, n, 0)
    american_put = american_binomial(False, S_0, K, T, r, delta, sigma, n, 0)

    print("Price of an american call with the given parameters: %f" % american_call)
    print("Price of an american put with the given parameters: %f" % american_put)