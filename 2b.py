import numpy as np


# Parameters
S_0 = 100  # Stock price
K = 100  # Strike
T = 3  # Maturity time
r = 0.04  # Risk-free rate
delta = 0.02  # Dividend yield
sigma = 0.2  # volatility
n = 500  # Number of binomial steps


def nCr(n_, r_):
    return np.math.factorial(n_) / (np.math.factorial(r_) * np.math.factorial(n_ - r_))


def european_binomial(call):
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
    for i in range(n):

        # Probability for ending up at node with i down moves
        prob = np.power(pu, n - i) * np.power(pd, i) * nCr(n, i)

        if call:
            # Intrinsic value of call at i down moves
            int_val = np.max([S_0 * np.power(u, n - i) * np.power(d, i) - K, 0])

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


euro_call = european_binomial(True)
euro_put = european_binomial(False)

print("Price of a european call with the parameters at the top: %f" % euro_call)
print("Price of a european put with the parameters at the top: %f" % euro_put)
