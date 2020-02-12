#########################################
# Matt Adiletta  - mjadiletta@wpi.edu   #
#########################################
# CS 541                                #
# HW3 Q 1                               #
# homework4_mjadiletta.py               #
#########################################
import numpy as np
import copy
import matplotlib.pyplot as plt

def randomize_training_order(Xtilde, y):
    ids = np.random.permutation(range(len(y)))
    Xtilde = Xtilde[:,ids]
    y = y[ids]
    return Xtilde, y

# unregularized fCE
def fCE (w, X, y):
    yhat = forward_propogation(X, w)[0]
    return -1/2*np.average(np.sum(y * np.log(yhat), axis=1))

def calc_yhat(mini_batch, w):
    z = np.dot(mini_batch.T, w)
    sum_row = np.asarray([np.sum(np.exp(z), axis=1)]*10).T
    yhat = np.divide(np.exp(z), sum_row)
    return yhat

def softmax(z):
    sum_row = np.asarray([np.sum(np.exp(z.T), axis=1)] * 10).T
    yhat = np.divide(np.exp(z.T), sum_row)
    return yhat

def forward_propogation(X, nn):
    z_intermediates = []
    h_intermediates = [X]
    h_out = X
    for w, b in nn:
        z_in = np.dot(w.T, h_out) + b[:, np.newaxis]
        z_intermediates.append(z_in)
        h_out = relu(z_in)
        h_intermediates.append(h_out)
    z_intermediates = z_intermediates[:-1]
    h_intermediates = h_intermediates[:-1]
    y = softmax(z_in)
    return y, z_intermediates, h_intermediates

def back_propogation(nn, y, ytr, z, h, lr, rs):
    nn_new = np.asarray(copy.deepcopy(nn))
    bp = y-ytr
    for i in reversed(range(len(nn))):
        nn_new[i][0] = nn[i][0] - lr * (np.dot(h[i], bp)/np.shape(h[i])[0] + rs*nn[i][0]) # check out division by hidden layer size
        nn_new[i][1] = nn[i][1] - lr * np.average(bp, axis=0) # check this for correctness
        if i != 0:
            bp = (np.dot(nn[i][0], bp.T)*grad_relu(z[i-1])).T
    return nn_new

def relu(X):
   return np.maximum(0, X)

def grad_relu(X):
    return np.ones(np.shape(X))

def stochastic_gradient_descent(X_tr, ytr, NETWORK_SHAPE, EPOCHS, MINIBATCH_SIZE, LEARNING_RATE, REGULARIZATION_STRENGTH):
    # NETWORK_SHAPE: [30, 40, 30] -> 3 layer network with 30 nodes in first layer, 40 in second, and 30 in third
    nn = list()
    NETWORK_SHAPE = copy.deepcopy(NETWORK_SHAPE)
    NETWORK_SHAPE.append(np.shape(ytr.T)[0])

    c = NETWORK_SHAPE[0] ** (-0.5)
    nn.append((np.asarray(np.random.uniform(low=(-c/2), high=(c/2), size=(np.shape(X_tr)[0], NETWORK_SHAPE[0]))), np.asarray([.01] * NETWORK_SHAPE[0]).T))
    for i in range(len(NETWORK_SHAPE)-1):
        c = NETWORK_SHAPE[i] ** (-0.5)
        nn.append((np.asarray(np.random.uniform(low=(-c/2), high=(c/2), size=(NETWORK_SHAPE[i], NETWORK_SHAPE[i+1]))), np.asarray([.01]*NETWORK_SHAPE[i+1]).T)) #np.asarray([np.random.randn(NETWORK_SHAPE[i+1])])

    for i in range(EPOCHS):
        for j in range(int(len(X_tr.T)/MINIBATCH_SIZE)):
            xtr_mini = X_tr.T[j*MINIBATCH_SIZE:(j+1)*MINIBATCH_SIZE].T
            ytr_mini = ytr[j*MINIBATCH_SIZE:(j+1)*MINIBATCH_SIZE]

            y, z, h = forward_propogation(xtr_mini, nn)
            lr = LEARNING_RATE * (EPOCHS-i/2)/EPOCHS
            nn = back_propogation(nn, y, ytr_mini, z, h, lr, REGULARIZATION_STRENGTH)
        if i % 1 == 0:
            print("\tEpoch:", i, "\tPercent Accuracy training set:", classify(nn, X_tr, ytr)*100)
    return nn

def classify(nn, X, y):
    difference = np.argmax(y, axis=1) == np.argmax(forward_propogation(X, nn)[0], axis=1)
    percent_accuracy = np.mean(difference)
    return percent_accuracy

def findBestHyperparameters(X_val, yval, optimal):
    NN_SHAPE = [[40], [200, 40], [200, 100, 40], [500, 200, 40]]

    EPOCHS = [20, 50, 100]  # Number of iterations over the training data
    MINIBATCH_SIZE = [32, 64, 128, 256]  # Number of minibatches
    LEARNING_RATE = [.05, .1]
    REGULARIZATION_STRENGTH = [0, .000005]

    print("Deriving optimal hyperparameters")
    min_fmse = float("inf")
    for nns in NN_SHAPE:
        for e in EPOCHS:
            for ms in MINIBATCH_SIZE:
                for lr in LEARNING_RATE:
                    for rs in REGULARIZATION_STRENGTH:
                        print("NNS:" + str(nns) + " E:" + str(e) + " MS:" + str(ms) + " LR:" + str(lr) + " RS:" + str(rs))
                        w = stochastic_gradient_descent(X_tr, ytr, nns, e, ms, lr, rs)
                        new_fmse = fCE(w, X_val, yval)
                        print("\t fCE: " + str(new_fmse))
                        if min_fmse > new_fmse:
                            optimal = {"NNS": nns, "E": e, "MB": ms, "LR": lr, "RS": rs}
                            print("new optimal" + str(optimal))
                            min_fmse = new_fmse
    print("Optimal Hyperparameters:")
    print("NNS:" + str(nns) + " E:" + str(e) + " MS:" + str(ms) + " LR:" + str(lr) + " RS:" + str(rs))
    return optimal

def train_mnist_softmax_regressor (X_tr, ytr, X_val, yval, derive_optimal_hp=False):
    X_tr, ytr = randomize_training_order(X_tr, ytr)

    optimal = {"NNS": [200, 40], "E": 100, "MB": 64, "LR": .5, "RS": .0000005}

    if(derive_optimal_hp):
        optimal = findBestHyperparameters(X_val, yval, optimal)
    return stochastic_gradient_descent(X_tr, ytr, optimal['NNS'], optimal['E'], optimal['MB'], optimal['LR'], optimal['RS']) #----------------Problem-----------------------------


if __name__ == "__main__":
    # Load data
    X_tr = np.load("mnist_train_images.npy").T
    ytr = np.load("mnist_train_labels.npy")
    X_val =  np.load("mnist_validation_images.npy").T
    yval = np.load("mnist_validation_labels.npy")
    X_te = np.load("mnist_test_images.npy").T
    yte = np.load("mnist_test_labels.npy")

    nn = train_mnist_softmax_regressor(X_tr, ytr, X_val, yval, True)

    print("Percent Accuracy training set:", classify(nn, X_tr, ytr))
    print("fCE training set:", fCE(nn, X_tr, ytr))
    print("Percent Accuracy validation set:", classify(nn, X_val, yval))
    print("fCE validation set:", fCE(nn, X_val, yval))
    print("Percent Accuracy testing set:", classify(nn, X_te, yte))
    print("fCE testing set:", fCE(nn, X_te, yte))

