class GraphPartial:

    def __init__(self, ins, op, out, grad_fn):
        self.ins = ins
        self.op = op
        self.out = out

        self.grad_fn = grad_fn