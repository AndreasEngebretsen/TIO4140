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
    for n in range(1, 501):
        call_option_values.append(european_binomial(True, S_0, K, T, r, delta, sigma, n))
        put_option_values.append(european_binomial(False, S_0, K, T, r, delta, sigma, n))
        x_axis.append(n)

    fig, axs = plt.subplots(2)
    fig.suptitle("Option prices as a function of binomial steps")

    axs[0].plot(x_axis, call_option_values)
    axs[0].set(xlabel="Number of binomial steps", ylabel="Call option prices")

    axs[1].plot(x_axis, put_option_values)
    axs[1].set(xlabel="Number of binomial steps", ylabel="Put option prices")

    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    call_val_1_10_100_500 = [call_option_values[0], call_option_values[9], call_option_values[99],
                             call_option_values[499]]
    put_val_1_10_100_500 = [put_option_values[0], put_option_values[9], put_option_values[99],
                            put_option_values[499]]

    return call_val_1_10_100_500, put_val_1_10_100_500


def task_2b_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    n = 100  # Number of binomial steps

    euro_call = european_binomial(True, S_0, K, T, r, delta, sigma, n)
    euro_put = european_binomial(False, S_0, K, T, r, delta, sigma, n)

    print("Price of a european call with the given parameters: %f" % euro_call)
    print("Price of a european put with the given parameters: %f" % euro_put)
