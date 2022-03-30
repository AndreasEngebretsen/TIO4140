import numpy as np


def simulate_stock_price_paths(S_0, steps, h, alpha, delta, sigma, simulations):
    all_stock_paths = []
    for sim in range(simulations):
        path = [S_0]
        S_h = S_0
        for i in range(steps):
            z = np.random.normal()
            S_h = S_h * np.exp((alpha - delta - 0.5 * np.power(sigma, 2)) * h + sigma * np.sqrt(h) * z)
            path.append(S_h)
        all_stock_paths.append(path)

    return all_stock_paths


def intrinsic_value(K, S_0, Call):
    if Call:
        return np.max([0, S_0 - K])
    else:
        return np.max([0, K - S_0])


def backward_induction_regression(stock_values, Call, K, r):
    no_paths = len(stock_values)
    time_steps = len(stock_values[0])

    cash_flow = {}
    for t in range(time_steps - 1, 1, -1):

        if t not in cash_flow:
            # Calculate intrinsic value for each path at time t
            cash_flow_time_t = []
            for path in range(no_paths):
                int_value = intrinsic_value(K, stock_values[path][t], Call)
                cash_flow_time_t.append(int_value)
            cash_flow[t] = cash_flow_time_t

        # Find values used in the regression
        x = []
        y = []
        for path in range(no_paths):

            # Calculate the intrinsic value of each path at time t - 1
            intrinsic_value_time_t_minus_one = intrinsic_value(K, stock_values[path][t - 1], Call)

            # If the intrinsic value at time t - 1 > 0, we include this path in the regression
            if intrinsic_value_time_t_minus_one > 0:

                # Add the stock price as the x value in the regression
                x.append(stock_values[path][t - 1])

                # Find non-zero discounted cash flow at current path to use as y value in regression
                y.append(0)
                for i in range(time_steps - 1, t - 1, -1):
                    if cash_flow[i][path] > 0:
                        y[-1] = (cash_flow[i][path] * np.exp(-r * (i - (t - 1))))

        # Calculate regression model used for finding continuation value
        coeffs = np.polyfit(x, y, 2)
        coeffs = [-1.813, 2.983, -1.070]

        # Compare the intrinsic value at time t-1 to the continuation value
        cash_flow_time_t_minus_one = []
        for path in range(no_paths):
            stock_value = stock_values[path][t - 1]
            if stock_value not in x:
                cash_flow_time_t_minus_one.append(0)
                continue
            continuation_value = coeffs[2] + coeffs[1] * stock_value + coeffs[0] * stock_value ** 2
            intrinsic_value_time_t_minus_one = intrinsic_value(K, stock_values[path][t - 1], Call)

            # If the intrinsic value at time t-1 is greater than the cash flow form the path, add it to cash flow
            if intrinsic_value_time_t_minus_one > continuation_value:
                cash_flow_time_t_minus_one.append(intrinsic_value_time_t_minus_one)

                # Set cash flow for time t equal to zero
                cash_flow[t][path] = 0
            else:
                # If cash flow at time t-1 is less than the continuation value, set cash flow at time t-1 equal to 0
                cash_flow_time_t_minus_one.append(0)
        cash_flow[t - 1] = cash_flow_time_t_minus_one

    return cash_flow


def value_option(cash_flow, r):
    option_value = 0
    for t in range(1, len(cash_flow) + 1):
        time_t_average = sum(cash_flow[t]) / len(cash_flow[t])
        option_value += time_t_average * np.exp(-r * t)

    return option_value


def american_monte_carlo(simulations, steps, S_0, T, alpha, delta, sigma, K):
    h = T / steps

    # Run simulations for stock paths
    all_stock_paths = simulate_stock_price_paths(S_0, steps, h, alpha, delta, sigma, simulations)

    # Calculate cash flows using LSM
    put_cash_flow = backward_induction_regression(all_stock_paths, False, K, alpha)
    call_cash_flow = backward_induction_regression(all_stock_paths, True, K, alpha)

    # Calculate option value based on calculated discounting cash flow and optimal exercise time
    put_option_value = value_option(put_cash_flow, alpha)
    call_option_value = value_option(call_cash_flow, alpha)

    return call_option_value, put_option_value


def task_3b_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    simulations = 10000  # Number of simulations
    steps = 50  # number of times exercisable each year

    call, put = american_monte_carlo(simulations, steps, S_0, T, r, delta, sigma, K)

    print(call, put)


task_3b_answers()
