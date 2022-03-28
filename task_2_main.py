from Task_2.task_2c import task_2c_answers
from Task_2.task_2b import task_2b_answers, binomial_steps_convergence
from Task_2.task_2d import task_2d_answers

call_vals, put_vals = binomial_steps_convergence()

task_2b_answers()
up_and_in_price = task_2c_answers()
up_and_out_price = task_2d_answers()

print("The sum of the up-and-out and the up-and-in call is: %f" % (up_and_in_price + up_and_out_price))
print(call_vals)
print(put_vals)