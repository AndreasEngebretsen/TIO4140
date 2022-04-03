import numpy as np
import matplotlib.pyplot as plt


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
        print(t)

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
    exercise_time = np.zeros(len(cash_flow)+1)
    for path in range(len(cash_flow[1])):
        for t in range(1, len(cash_flow) + 1):
            option_value += cash_flow[t][path] * np.exp(-r * t)

            if cash_flow[t][path] > 0:
                exercise_time[t] += 1

    option_value = option_value / len(cash_flow[1])

    return option_value, exercise_time


def create_histogram(option_ex_lst, name):
    hist_data = []
    for i in range(len(option_ex_lst)):
        for h in range(int(option_ex_lst[i])):
            hist_data.append(i)

    plt.plot()
    plt.hist(hist_data, bins=10)
    plt.title(name)
    plt.ylabel("Number of options exercised at time t")
    plt.ylabel("t")
    plt.show()


def american_monte_carlo(simulations, steps, S_0, T, alpha, delta, sigma, K, Call):
    h = T / steps

    # Run simulations for stock paths
    all_stock_paths = simulate_stock_price_paths(S_0, steps, h, alpha, delta, sigma, simulations)

    # all_stock_paths = [[1, 1.09, 1.08, 1.34],
    #                   [1, 1.16, 1.26, 1.54],
    #                  [1, 1.22, 1.07, 1.03],
    #                  [1, 0.93, 0.97, 0.92],
    #                  [1, 1.11, 1.56, 1.52],
    #                  [1, 0.76, 0.77, 0.90],
    #                  [1, 0.92, 0.84, 1.01],
    #                  [1, 0.88, 1.22, 1.34]]

    # Calculate cash flows using LSM
    put_cash_flow = backward_induction_regression(all_stock_paths, False, K, alpha)
    call_cash_flow = 0
    if Call:
        call_cash_flow = backward_induction_regression(all_stock_paths, True, K, alpha)

    # Calculate option value based on calculated discounting cash flow and optimal exercise time
    put_option_value, put_exercise_time = value_option(put_cash_flow, alpha)
    call_option_value = 0
    call_exercise_time = 0
    if Call:
        call_option_value, call_exercise_time = value_option(call_cash_flow, alpha)

    return call_option_value, put_option_value, call_exercise_time, put_exercise_time


def task_3b_answers():
    # Parameters
    S_0 = 1  # Stock price
    K = 1.10  # Strike
    T = 3  # Maturity time
    r = 0.06  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # volatility
    simulations = 1  # Number of simulations
    steps = 3  # number of times exercisable each year

    call, put, c_ex, p_ex = american_monte_carlo(simulations, steps, S_0, T, r, delta, sigma, K, True)

    print("American monte carlo call value: %f, American monte carlo put value: %f" % (call, put))


def task_3c_answers():
    # Parameters
    S_0 = 1  # Stock price
    K = 1.10  # Strike
    T = 3  # Maturity time
    r = 0.06  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.2  # Volatility
    simulations = 10000  # Number of simulations
    steps = 50  # number of times exercisable each year

    call, put, c_ex, p_ex = american_monte_carlo(simulations, steps, S_0, T, r, delta, sigma, K, True)

    ex_before_mat_1 = np.sum(p_ex[:(len(p_ex)-1)]) / np.sum(p_ex)

    print("-----------------------------------------------------")
    print("Task 3 c)")
    print("Percentage of paths where the option is exercised: %f" % ex_before_mat_1)

    create_histogram(p_ex)


def task_3d_answers():
    # Parameters
    S_0 = 1  # Stock price
    K = 1.10  # Strike
    T = 3  # Maturity time
    r = 0.06  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma_1 = 0.1  # Volatility
    sigma_2 = 0.4  # Volatility
    simulations = 10000  # Number of simulations
    steps = 50  # number of times exercisable each year

    call_1, put_1, c_ex_1, p_ex_1 = american_monte_carlo(simulations, steps, S_0, T, r, delta, sigma_1, K, False)
    call_2, put_2, c_ex_2, p_ex_2 = american_monte_carlo(simulations, steps, S_0, T, r, delta, sigma_2, K, False)

    print("-----------------------------------------------------")
    print("Task 3 d)")

    create_histogram(p_ex_1, "Sigma = %f" % sigma_1)
    create_histogram(p_ex_2, "Sigma = %f" % sigma_2)



