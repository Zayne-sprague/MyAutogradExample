from nn.module.Module import Module
from nn.module.Linear import Linear
from nn.losses.NLL import NLL
from nn.activations.logsoftmax import LogSoftmax
from nn.activations.sigmoid import Sigmoid
from nn.activations.relu import ReLU

from data_loader import *

from zensor import Zensor as Z
from zensor.Zensor import zArgmax

import numpy as np

from tqdm import tqdm

class FFN_MINST(Module):

    def __init__(self):
        super().__init__()

        self.FC = Linear(784, 100)
        self.FC_A = ReLU()

        self.FC2 = Linear(100, 10)
        self.L_SOFTMAX = LogSoftmax()

    def forward(self, x):

        x = self.FC(x)
        x = self.FC_A(x)
        x = self.FC2(x)
        x = self.L_SOFTMAX(x)

        return x

    @staticmethod
    def clip_grad(grad):
        g = grad / 128.
        s = np.sqrt(np.sum(np.square(g)))
        if s > 5.:
            g = g * 5. / s
        return g


    def optim(self, lr):

        self.FC.weights.requires_grad = False
        self.FC.bias.requires_grad = False
        self.FC2.weights.requires_grad = False
        self.FC2.bias.requires_grad = False

        self.FC.weights -= self.clip_grad(self.FC.weights.grad) * lr
        self.FC.bias -= self.clip_grad(self.FC.bias.grad) * lr

        self.FC2.weights -= self.clip_grad(self.FC2.weights.grad) * lr
        self.FC2.bias -= self.clip_grad(self.FC2.bias.grad) * lr

        self.FC.weights.requires_grad = True
        self.FC.bias.requires_grad = True
        self.FC2.weights.requires_grad = True
        self.FC2.bias.requires_grad = True

        self.clear_weight_grads()
        return

    def clear_weight_grads(self):
        self.FC.weights.zero_grad()
        self.FC.bias.zero_grad()
        self.FC2.weights.zero_grad()
        self.FC2.bias.zero_grad()

    def get_weights(self):
        return self.FC.weights.value.copy(), self.FC.bias.value.copy(), self.FC2.weights.value.copy(), self.FC2.bias.value.copy()

    def set_weights(self, weights):
        FCW, FCB, FC2W, FC2B = weights
        self.FC.weights = Z(FCW)
        self.FC.bias = Z(FCB)
        self.FC2.weights = Z(FC2W)
        self.FC2.bias = Z(FC2B)


def accuracy(out, yb):
    preds = zArgmax(out, axis=1)
    return (preds == yb.value.tolist()).float().mean()

def train(model: FFN_MINST, x_train, y_train, batch_size, epochs, learning_rate, L2_alpha=0.002 ):

    criterion = NLL()

    best_weights = None

    best_acc = -1
    best_loss = -1

    n = len(x_train)

    for epoch in range(epochs):

        rng = range((n - 1) // batch_size + 1)
        p_bar = tqdm(rng, desc="TRAINING")
        acc = 0
        epoch_loss = 0

        for i in p_bar:
            #         set_trace()
            start_i = i * batch_size
            end_i = min(n, start_i + batch_size)

            if end_i - start_i < batch_size:
                continue

            xb = Z(x_train[start_i:end_i])
            yb = Z(y_train[start_i:end_i])


            model.clear_weight_grads()
            pred = model(xb)
            loss = criterion(pred, yb) + (model.FC.weights.square().sum().sum().sqrt() + model.FC.bias.square().sum().sqrt() + model.FC2.weights.square().sum().sum().sqrt() + model.FC2.bias.square().sum().sqrt()) * L2_alpha


            loss.backward()

            epoch_loss += loss.value / len(rng)
            acc += accuracy(pred, yb).value / len(rng)

            model.optim(learning_rate)


            p_bar.set_description(f"TRAINING EPOCH: {epoch + 1}, LOSS = {epoch_loss} BEST LOSS = {best_loss}, ACC = {acc} BEST_ACC = {best_acc}")

        if best_acc < acc or best_acc == -1:
            best_acc = acc.copy()
            best_loss = loss.value.copy()

            best_weights = model.get_weights()

    model.set_weights(best_weights)
    return model

def eval(model: FFN_MINST, x_data, y_data, batch_size):

    total_loss = 0
    total_acc = 0
    total_n = 0

    criterion = NLL()

    n = len(x_data)

    for i in range((len(x_data) - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = min(n, start_i + batch_size)

        if end_i - start_i < batch_size:
            continue

        xb = Z(x_data[start_i:end_i])
        yb = Z(y_data[start_i:end_i])

        pred = model(xb)

        ls = criterion(pred, yb).value
        acc = accuracy(pred, yb).value

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

    return total_loss, total_acc


if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid = get_examples()
    n = len(x_train)

    model = FFN_MINST()

    model = train(model, x_train, y_train, 32, 15, 1, L2_alpha=0.01)
    eval(model, x_valid, y_valid, 32)

