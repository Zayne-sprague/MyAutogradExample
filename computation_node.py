class ComputationNode:

    def __init__(self, zensor_node, operation_lambda, op=None):
        self.input_node = zensor_node
        self.grad_func = operation_lambda

        if op:
            self.op = op

