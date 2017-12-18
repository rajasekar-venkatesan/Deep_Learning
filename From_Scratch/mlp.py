#imports
import numpy as np
import time
from mlp_functions import *

x_list = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
y_list = [[[0]], [[1]], [[1]], [[0]]]

x = [variable(x_item) for x_item in x_list]
y = [variable(y_item) for y_item in y_list]

hidden_size = 6
num_epochs = 500
lrate = 0.1

i2h_list = np.random.randn(x[0].data.shape[1], hidden_size)
i2h = variable(i2h_list)

h2o_list = np.random.randn(hidden_size, y[0].data.shape[1])
h2o = variable(h2o_list)
p_average_loss = 100

for e in range(num_epochs + 1):
    start_time = time.time()
    costs = []
    times = []
    for x_train, y_train in zip(x, y):
        h_in = matmul(x_train, i2h)
        h_out = sigmoid_activation(h_in.out_var)
        f_in = matmul(h_out.out_var, h2o)
        f_out = sigmoid_activation(f_in.out_var)
        err = sub_matrix(f_out.out_var, y_train)
        cost = square(err.out_var)

        costs.append(cost.out_var.data[0][0])

        final_grad(cost.out_var)

        grad_back_prop()

        h2o.data = h2o.data - (h2o.grad * lrate)
        i2h.data = i2h.data - (i2h.grad * lrate)

        del_grads()
        del_op_stack()
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    if e % 10 == 0:
        average_loss = sum(costs)/len(costs)
        average_time = sum(times)/len(times)
        print('Cost at epoch {0:4d} is {1:.5f}. Time/epoch: {2:.5f}s'.format(e, round(average_loss, 5), round(average_time, 5)))
        if average_loss < 1e-5:
            print('==> Loss is less than 10^(-5). Training Stopped.')
            break
        if abs(p_average_loss - average_loss) < 1e-5:
            print('==> Loss convergence is less than 10^(-5). Training Stopped.')
            break

        costs = []
        times = []

y_pred = []
for x_train, y_train in zip(x, y):
    h_in = matmul(x_train, i2h)
    h_out = sigmoid_activation(h_in.out_var)
    f_in = matmul(h_out.out_var, h2o)
    f_out = sigmoid_activation(f_in.out_var)
    y_pred.append(np.round(f_out.out_var.data, 2).tolist())

print('Actual Output: {}'.format(y_list))
print('Predicted Output: {}'.format(y_pred))

