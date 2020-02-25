import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

class RNN:
    def __init__(self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1

        self.h_past = None

        self.h_history = []  # in time steps
        self.g_history = []  # in time steps

    def tanh(self, val):
        return np.tanh(val)

    def g_tanh(self, val): # val is numHidden x 1
        return np.array(1-self.tanh(val)**2).flatten()

    def backward(self, yhat, x, y, lr):
        dJ_dU = []
        dJ_dV = []
        dJ_dW = []
        dJ_dy_hat = (yhat-y)
        dy_hat_dh = self.w.T

        for t in reversed(range(len(y))): # dJ/dU_t ...
            g_t = self.g_history[t]
            q_prev = np.dot(dJ_dy_hat[t], dy_hat_dh) * g_t

            r_u_reversed = []
            r_v_reversed = []
            for tau in range(t-1):
                r_u_reversed.append(np.outer(q_prev, self.h_history[t-tau-1]))
                r_v_reversed.append(np.outer(q_prev, x[t-tau]))
                q_prev = np.dot(q_prev, self.U) * self.g_history[t-tau-1]

            dJ_dU.append(np.sum(r_u_reversed, axis=0))
            dJ_dV.append(np.sum(r_v_reversed, axis=0))
            dJ_dW.append(np.dot(dJ_dy_hat[t], self.h_history[t]))

        return np.sum(dJ_dU, axis=0), np.sum(dJ_dV, axis=0), np.sum(dJ_dW, axis=0) # grad_u, grad_v, grad_w

    def forward(self, xs):
        y_hats = np.zeros(np.shape(xs))
        for k, x in enumerate(xs):
            z_t = np.dot(self.U, self.h_past) + self.V*x
            h_new = self.tanh(z_t)
            self.h_history.append(h_new)
            self.g_history.append(self.g_tanh(z_t))
            y_hats[k] = np.dot(h_new.T, self.w)
            self.h_past = h_new
        return y_hats

    def accuracy(self, x, y):
        return np.sum(.5*(self.forward(x) - y) ** 2)

def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    np.random.seed(3)

    EPOCHS = 100000
    LEARNING_RATE = 0.001
    LEARNING_RATE_W = 0.1
    RS = .0001
    CLIP_MAX = 100

    xs, ys = generateData()

    numHidden = 6
    numInput = 1

    rnn = RNN(numHidden, numInput, 1)

    for i in range(EPOCHS):
        rnn.h_past = np.zeros((numHidden, numInput))
        rnn.g_history = []
        rnn.h_history = []

        y_hats = rnn.forward(xs)
        ret = rnn.backward(y_hats, xs, ys, LEARNING_RATE)

        grad_u = np.clip(ret[0], a_min=-CLIP_MAX, a_max=CLIP_MAX)
        grad_v = np.clip(ret[1], a_min=-CLIP_MAX, a_max=CLIP_MAX)
        grad_w = np.clip(ret[2], a_min=-CLIP_MAX, a_max=CLIP_MAX)

        lr = LEARNING_RATE * (EPOCHS - i)/EPOCHS
        lr_w = LEARNING_RATE_W * (EPOCHS - i) / EPOCHS

        rnn.U = rnn.U - lr * (grad_u + RS*rnn.U)
        rnn.V = rnn.V - lr * (grad_v + RS*rnn.V)
        rnn.w = rnn.w - lr_w * grad_w.flatten()

        xs_t , ys_t = generateData()
        accuracy = rnn.accuracy(xs_t, ys_t)
        print("\tEpoch:", i, "\tLoss testing:", accuracy)
        if accuracy < .05:
            print("Accuracy Achieved!")
            print("\tEpoch:", i, "\tLoss training:", rnn.accuracy(xs, ys))
            print("\tEpoch:", i, "\tLoss testing:", accuracy)
            exit()

    print("Accuracy not reached!")
