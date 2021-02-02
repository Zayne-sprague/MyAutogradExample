from zensor import *
from data_loader import *

# TODO - NOT WORKING AS OF REFACTOR

class FNN:

    def __init__(self, input, hidden_size, hidden_layers, out):
        if not hidden_layers or hidden_layers == 0:
            hidden_size = out

        self.input = Zensor(np.random.rand([input, hidden_size]))
        self.in_bias = Zensor(np.zeros([input, 1]))

        self.hiddens = []
        self.hidden_biases = []
        for i in range(0, hidden_layers):
            self.hiddens.append(Zensor(np.random.rand([hidden_size, hidden_size])))
            self.hidden_biases.append(Zensor(np.zeros([hidden_size, 1])))



        self.output = Zensor(np.random.rand([hidden_size, out]))
        self.out_bias = Zensor(np.zeros([out, 1]))


    def predict(self, inputs):

        out = self.input @ inputs + self.in_bias
        for i in range(0, len(self.hiddens)):
            out = out @ self.hiddens[i] + self.hidden_biases[i]

        out = out @ self.output + self.out_bias
        out = zensor_Sigmoid(out)

        return out


if __name__ == "__main__":
    import math
    x_train, y_train, x_valid, y_valid = get_examples()
    n = len(x_train)

    weights = Zensor(np.random.rand(784, 10) / math.sqrt(784))
    biases = Zensor(np.zeros([10]))

    def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)

    def model(xb):
        return log_softmax(xb @ weights + biases)

    def nll(ins, target):
        return -ins[range(target.shape(0)[0]), target.value].mean()

    def accuracy(out, yb):
        preds = zensor_argmax(out, axis=1)
        return (preds == yb.value.tolist()).float().mean()

    loss_func = nll

    bs = 512  # batch size

    # xb = Zensor(x_train[0:bs])
    # preds = model(xb)  # predictions

    # yb = Zensor(y_train[0:bs])

    # print(loss_func(preds,  yb))

    # print(accuracy(preds, yb))

    lr = 0.00005  # learning rate
    epochs = 2  # how many epochs to train for

    best_acc = 0
    best_acc_loss = 0
    best_w = Zensor(weights.value.copy())
    best_b = Zensor(biases.value.copy())

    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = Zensor(x_train[start_i:end_i])
            yb = Zensor(y_train[start_i:end_i])
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()

            weights.require_grad = False
            biases.require_grad = False

            ls = loss_func(model(xb), yb)
            acc = accuracy(model(xb), yb).value

            if acc > best_acc:
                best_acc = acc
                best_acc_loss = ls.value
                best_w = Zensor(weights.value.copy())
                best_b = Zensor(biases.value.copy())

            weights.value += weights.grad * lr
            biases.value += biases.grad * lr

            weights = weights.zero_grad()
            biases = biases.zero_grad()

            print(ls, acc)

            weights.require_grad = True
            biases.require_grad = True



    total_loss = 0
    total_acc = 0
    total_n = 0

    weights = best_w
    biases = best_b

    weights.require_grad = False
    biases.require_grad = False

    x_data = x_valid
    y_data = y_valid

    for i in range((len(x_data) - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = Zensor(x_data[start_i:end_i])
        yb = Zensor(y_data[start_i:end_i])

        pred = model(xb)
        ls = loss_func(pred, yb).value
        acc = accuracy(model(xb), yb).value

        if np.isnan(ls):
            ls = 0.0
        if np.isnan(acc):
            acc = 0.0
        total_loss += ls
        total_acc += acc
        total_n += 1

    total_acc /= total_n
    total_loss /= total_n

    print ("--- BEST MODEL VALIDATIONS STATS ---")
    print(f"\tLOSS: {total_loss}")
    print(f"\tACC:  {total_acc}")
    print(f"\n--- TRAINING STATS ---\n\tLOSS: {best_acc_loss}\n\tACC:  {best_acc}")

