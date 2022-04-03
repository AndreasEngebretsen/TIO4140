from Task_2.task_2c import task_2c_answers
from Task_2.task_2b import task_2b_answers, binomial_steps_convergence
from Task_2.task_2d import task_2d_answers

call_values, put_values = binomial_steps_convergence()

task_2b_answers()
up_and_in_price = task_2c_answers()
up_and_out_price = task_2d_answers()

print("The sum of the up-and-out and the up-and-in call is: %f" % (up_and_in_price + up_and_out_price))
print("1 binomial step. Call: %f, Put: %f" % (call_values[0], put_values[0]))
print("50 binomial step. Call: %f, Put: %f" % (call_values[1], put_values[1]))
print("100 binomial step. Call: %f, Put: %f" % (call_values[2], put_values[2]))
print("500 binomial step. Call: %f, Put: %f" % (call_values[3], put_values[3]))
