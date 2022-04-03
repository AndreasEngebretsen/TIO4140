import warnings
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


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


# Function to simulate the stock paths
def simulate_stock_paths(S_0, steps, h, alpha, delta, sigma, simulations):
    # All the stock paths
    all_stock_paths = []

    # Generate all the simulations
    for i in range(simulations):

        # Each simulation starts with stock price equal to S_0
        path = [S_0]
        S_h = S_0

        # Each simulation need to simulate all the steps
        for t in range(1, steps):
            # Generate a number from the standard normal distribution
            z = np.random.normal()

            # Generate the new stock value
            s_t = S_h * np.exp(((alpha - delta) - 0.5 * (sigma ** 2) * h) + sigma * np.sqrt(h) * z)
            path.append(s_t)
        all_stock_paths.append(path)

    # Return the stock paths as an np array
    return np.array(all_stock_paths)


# Function to calculate the intrinsic values of the options in all the paths if they were exercised
def intrinsic_values(stock_paths, call, K):
    intrinsic_values_return = []
    for row in stock_paths:
        row_with_intrinsic_values = []
        for stock_price in row:
            if call:
                row_with_intrinsic_values.append(max(stock_price - K, 0))
            else:
                row_with_intrinsic_values.append(max(K - stock_price, 0))
        intrinsic_values_return.append(row_with_intrinsic_values)
    intrinsic_values_return = np.array(intrinsic_values_return)
    return intrinsic_values_return


def values_at_step_n(paths, n):
    values = []
    for row in paths:
        values.append(row[n])
    return values


def backward_induction_regression(steps, stock_paths, intrinsic_values_, r, h):
    int_values = np.array(deepcopy(intrinsic_values_))
    cash_flow = deepcopy(intrinsic_values_[:, -1])
    execution_matrix = np.zeros((len(intrinsic_values_), len(intrinsic_values_[0])))
    itm_last_period = deepcopy(int_values[:, -1])
    itm_last_period = np.divide(itm_last_period, itm_last_period)
    itm_last_period[np.isnan(itm_last_period)] = 0
    execution_matrix[:, -1] = itm_last_period

    # Starting at the back, we calculate
    for n in range(steps - 1, 1, -1):

        cash_flow = cash_flow * np.exp(-r * h)

        itm = int_values[:, n - 1]
        itm = np.divide(itm, itm)
        itm[np.isnan(itm)] = 0

        # x is the stock values at step n - 1
        x = deepcopy(stock_paths[:, n - 1])
        x = x * itm
        x = x[x > 0]

        # y is the intrinsic values at step n
        y = deepcopy(cash_flow)
        y[y == 0] = -1
        y = y * itm
        y = y[y != 0]
        y[y < 0] = 0

        if not y.size > 0:
            continue
        # Calculate the regression
        regression = np.polyfit(x, y, 2)

        # The intrinsic values at step n minus one
        intrinsic_values_at_step_n_minus_one = int_values[:, n - 1]
        intrinsic_values_at_step_n_minus_one[intrinsic_values_at_step_n_minus_one == 0.0] = -1
        intrinsic_values_at_step_n_minus_one *= itm
        intrinsic_values_at_step_n_minus_one = intrinsic_values_at_step_n_minus_one[
            intrinsic_values_at_step_n_minus_one != 0]
        intrinsic_values_at_step_n_minus_one[intrinsic_values_at_step_n_minus_one < 0] = 0

        # Going through all the intrinsic values at step n - 1 and checking if
        for k in range(len(x)):
            cont_val = np.polyval(regression, x[k])
            int_val = intrinsic_values_at_step_n_minus_one[k]

            if int_val > cont_val:
                itm_index_lst = np.where(itm == 1)
                itm_index = itm_index_lst[0][k]
                cash_flow[itm_index] = int_val
                execution_matrix[itm_index][n - 1] = 1
                execution_matrix[itm_index][n] = 0

    cash_flow *= np.exp(-r * h)
    executed = execution_matrix.sum(axis=0)

    return cash_flow, executed


# Function to calculate the American ls monte carlo value of a put and a call
def american_ls_monte_carlo(T, r, steps, K, S_0, delta, sigma, simulations, Call):
    h = T / steps

    # Calculate all the stock paths
    stock_paths = simulate_stock_paths(S_0, steps, h, r, delta, sigma, simulations)

    # Calculate the intrinsic values at all time steps
    put_values = intrinsic_values(stock_paths, False, K)
    call_values = 0
    if Call:
        call_values = intrinsic_values(stock_paths, True, K)

    # Backward induction regression
    put_cash_flow, put_execution_matrix = backward_induction_regression(steps, stock_paths, put_values, r, h)
    call_cash_flow = [1]
    call_execution_matrix = 0
    if Call:
        call_cash_flow, call_execution_matrix = backward_induction_regression(steps, stock_paths, call_values, r, h)

    # Divide the discounted cash flow in the number of paths to get the average
    avg_put_value = sum(put_cash_flow) / len(put_cash_flow)
    avg_call_value = 0
    if Call:
        avg_call_value = sum(call_cash_flow) / len(call_cash_flow)

    return avg_call_value, avg_put_value, call_execution_matrix, put_execution_matrix


def task_3b_answers():
    # Parameters
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma = 0.20  # volatility
    simulations = 10000  # Number of simulations
    steps = 50  # number of times exercisable each year

    call, put, c_ex, p_ex = american_ls_monte_carlo(T, r, steps, K, S_0, delta, sigma, simulations, True)

    print("American monte carlo call value: %f \n"
          "American monte carlo put value: %f" % (call, put))


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

    call, put, c_ex, p_ex = american_ls_monte_carlo(T, r, steps, K, S_0, delta, sigma, simulations, True)

    ex_before_mat_1 = np.sum(p_ex[:(len(p_ex)-1)]) / np.sum(p_ex)

    print("-----------------------------------------------------")
    print("Task 3 c)")
    print("Percentage of paths where the option is exercised: %f" % ex_before_mat_1)

    create_histogram(p_ex, "Exercised")


def task_3d_answers():
    S_0 = 100  # Stock price
    K = 100  # Strike
    T = 3  # Maturity time
    r = 0.04  # Risk-free rate
    delta = 0.02  # Dividend yield
    sigma_1 = 0.05  # volatility
    sigma_2 = 0.3
    simulations = 10000  # Number of simulations
    steps = 50  # number of times exercisable each year

    call, put, c_ex, p_ex_1 = american_ls_monte_carlo(T, r, steps, K, S_0, delta, sigma_1, simulations, False)
    call, put, c_ex, p_ex_2 = american_ls_monte_carlo(T, r, steps, K, S_0, delta, sigma_2, simulations, False)

    create_histogram(p_ex_1, "Sigma = %f" % sigma_1)
    create_histogram(p_ex_2, "Sigma = %f" % sigma_2)
