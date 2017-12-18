#imports
import numpy as np

#global variables
operations_stack = []
clear_grad_stack = []

#functions
def prep_data(in_data):
    if isinstance(in_data, list):
        out_data = np.array(in_data)
        return out_data
    elif isinstance(in_data, np.ndarray):
        out_data = in_data
        return out_data
    else:
        print('Input Data is of type {} which is not supported'.format(type(in_data)))
        return None

def grad_back_prop():
    for i in reversed(operations_stack):
        i()

def del_op_stack():
    global operations_stack
    operations_stack = []

def del_grads():
    global clear_grad_stack
    # for var in clear_grad_stack:
    #     var.grad = np.zeros_like(var.grad)
    clear_grad_stack = []

def softmax(in_var):
    exps = np.exp(in_var.data - np.max(in_var.data))
    out_var = exps / np.sum(exps)
    return out_var.tolist()


class variable():
    def __init__(self, data):
        #data - list or numpy array, self.data - numpy array
        self.data = prep_data(data)
        #gradient - initialized to zeros, self.grad - numpy array
        self.grad = np.zeros(self.data.shape)

    def read_data(self):
        return self.data

    def write_data(self, in_data):
        in_data = prep_data(in_data)
        if in_data.shape == self.data.shape:
            # gradient is maintained
            self.data = in_data
        else:
            # gradient is reinitialized to match the data shape
            self.data = in_data
            self.grad = np.zeros(self.data.shape)

    def read_grad(self):
        return self.grad

    def write_grad(self, in_grad):
        in_grad = prep_data(in_grad)
        if in_grad.shape == self.grad.shape:
            self.grad = in_grad
        else:
            print('Expected gradient of shape {} but received gradient of shape {}'.format(self.grad.shape, in_grad.shape))

    def accumulate_grad(self, in_grad):
        in_grad = prep_data(in_grad)
        if in_grad.shape == self.grad.shape:
            self.grad += in_grad
        else:
            print('Expected gradient of shape {} but received gradient of shape {}'.format(self.grad.shape, in_grad.shape))

    def clear_grad(self):
        self.grad = np.zeros(self.grad.shape)


class operation:
    def __init__(self, *in_vars):
        self.in_vars = [var for var in in_vars]
        self.out_var = None

    def backward(self):
        pass

class sum_matrix(operation):
    def __init__(self, *in_vars):
        super(sum_matrix, self).__init__(*in_vars)
        self.out_var = variable(np.zeros(self.in_vars[0].data.shape))

        #forward operation
        for in_var in self.in_vars:
            self.out_var.data += in_var.data

        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)


    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        for in_var in self.in_vars:
            in_var.accumulate_grad(self.out_var.grad)

class sub_matrix(operation):
    def __init__(self, *in_vars):
        super(sub_matrix, self).__init__(*in_vars)
        self.out_var = variable(np.zeros(self.in_vars[0].data.shape))

        #forward operation
        self.out_var.data = self.in_vars[0].data - self.in_vars[1].data

        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)


    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        for in_var in self.in_vars:
            in_var.accumulate_grad(self.out_var.grad)

class sum_elements(operation):
    def __init__(self, *in_vars):
        super(sum_elements, self).__init__(*in_vars)
        self.out_var = variable(np.zeros((len(self.in_vars),1)))

        #forward operation
        for i, in_var in enumerate(self.in_vars):
            self.out_var.data[i] += np.sum(in_var.data)

        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)


    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        for i, in_var in enumerate(self.in_vars):
            in_var.accumulate_grad(np.ones(in_var.grad.shape)*self.out_var.grad[i])


class square(operation):
    def __init__(self, *in_vars):
        super(square, self).__init__(*in_vars)
        self.out_var = variable(np.zeros(self.in_vars[0].data.shape))

        #forward operation
        self.out_var.data = np.square(self.in_vars[0].data)
        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)

    #backward operation
    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        in_var = self.in_vars[0]
        local_grad = 2*in_var.data
        in_var.accumulate_grad(local_grad * self.out_var.grad)

class sigmoid_activation(operation):
    def __init__(self, *in_vars):
        super(sigmoid_activation, self).__init__(*in_vars)
        self.out_var = variable(np.zeros(self.in_vars[0].data.shape))

        #forward operation
        self.out_var.data = 1 / (1 + np.exp(-self.in_vars[0].data))
        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)

    #backward operation
    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        in_var = self.in_vars[0]

        local_sig = 1 / (1 + np.exp(-self.in_vars[0].data))
        local_grad = local_sig * (1 - local_sig)
        in_var.accumulate_grad(local_grad * self.out_var.grad)

class matmul(operation):
    def __init__(self, *in_vars):
        super(matmul, self).__init__(*in_vars)
        self.out_var = variable(np.zeros((in_vars[0].data.shape[0], in_vars[1].data.shape[1])))

        #forward operation
        self.out_var.data = np.matmul(in_vars[0].data, in_vars[1].data)
        operations_stack.append(self.backward)
        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)

    #backward operation
    def backward(self, in_grad=None):
        if in_grad is not None:
            in_grad = prep_data(in_grad)
            if in_grad.shape == self.out_var.grad.shape:
                self.out_var.grad = in_grad
            else:
                print('!!! ==> Expected input gradient of shape {} but received gradient of shape {} \n!!! ==> Gradients not updated'.format(self.out_var.grad.shape, in_grad.shape))

        in_var0_grad = np.matmul(self.out_var.grad, self.in_vars[1].data.T)
        in_var1_grad = np.matmul(self.in_vars[0].data.T, self.out_var.grad)
        self.in_vars[0].accumulate_grad(in_var0_grad)
        self.in_vars[1].accumulate_grad(in_var1_grad)


class final_grad(operation):
    def __init__(self, *in_vars):
        super(final_grad, self).__init__(*in_vars)
        for in_var in in_vars:
            in_var.grad = np.ones(in_var.grad.shape)

        for in_var in self.in_vars:
            clear_grad_stack.append(in_var)

if __name__ == '__main__':
    pass