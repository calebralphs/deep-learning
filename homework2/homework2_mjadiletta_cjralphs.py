#########################################
# Matt Adiletta  - mjadiletta@wpi.edu   #
#########################################
# CS 541                                #
# HW1 Part 1                            #
# homework1_mjadiletta.py               #
#########################################
import numpy as np
import matplotlib.pyplot as plt

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    shape = np.shape(faces)
    new_faces = faces.reshape(shape[0], shape[1]*shape[2]).T
    ones = np.ones(shape[0])
    new_faces = np.vstack((new_faces, ones))
    return new_faces

def randomize_training_order(Xtilde, y):
    ids = np.random.permutation(range(len(y)))
    Xtilde = Xtilde[:,ids]
    y = y[ids]
    return Xtilde, y

# (x*x.T)*w = x*y -> np.linalg.solve returns w
def linear_regression (X_tr, ytr):
    return np.linalg.solve(np.dot(X_tr, X_tr.T), np.dot(X_tr, ytr))

# l2 regularized fMSE
def l2fMSE(w, X, y, alpha):
    yhat = np.dot(X.T, w)
    fmse = np.average((yhat - y) ** 2 + alpha/2*np.dot(w.T, w))/2
    return fmse

# unregularized fMSE
def fMSE (w, X, y):
    yhat = np.dot(X.T, w)
    fmse = np.average((yhat - y) ** 2)/2
    return fmse

def gradient(xtr_mini, y, w, l2reg):
    return np.dot(xtr_mini, (np.dot(xtr_mini.T, w)-y))/np.shape(xtr_mini)[0] + l2reg*w

def stochastic_gradient_descent(X_tr, ytr, EPOCHS, MINIBATCH_SIZE, LEARNING_RATE, REGULARIZATION_STRENGTH):
    w = np.asarray(.1 * np.random.randn(np.shape(X_tr)[0]))
    for i in range(EPOCHS):
        for j in range(int(len(X_tr.T)/MINIBATCH_SIZE)):
            xtr_mini = X_tr.T[j*MINIBATCH_SIZE:(j+1)*MINIBATCH_SIZE].T
            ytr_mini = ytr[j*MINIBATCH_SIZE:(j+1)*MINIBATCH_SIZE]
           grad_w = gradient(xtr_mini, ytr_mini, w, REGULARIZATION_STRENGTH)
            w = w - LEARNING_RATE * grad_w
    return w

def showImage(weights):
    # Show an arbitrary test image in grayscale
    weights = weights[:-1].reshape(48,48)
    fig,ax = plt.subplots(1)
    ax.imshow(weights, cmap='gray')
    plt.show()


def train_age_regressor (derive_optimal_hp=False):
    # Load data
    X_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    X_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    X_tr, ytr = randomize_training_order(X_tr, ytr)

    X_val = X_tr.T[:][0:int(np.shape(X_tr)[1]*.2)].T
    y_val = ytr.T[0:int(np.shape(X_tr)[1]*.2)].T
    X_tr = X_tr.T[:][int(np.shape(X_tr)[1]*.2):int(np.shape(X_tr)[1])].T
    ytr = ytr.T[int(np.shape(ytr)[0]*.2):int(np.shape(ytr)[0])].T

    EPOCHS = [500, 1000, 2000, 5000]  # Number of iterations over the training data
    MINIBATCH_SIZE = [50, 100, 500, 1000]  # Number of minibatches
    LEARNING_RATE = [.001, .01, .05, .1]
    REGULARIZATION_STRENGTH = [.0005, .001, .005, .01]

    min_fmse = float("inf")
    optimal = {"E": 5000, "MB": 500, "LR": .01, "RS": .001}
    # 2000, 500, .01, .001

    if derive_optimal_hp:
        "Deriving optimal hyperparameters"
        for e in EPOCHS:
            for ms in MINIBATCH_SIZE:
                for lr in LEARNING_RATE:
                    for rs in REGULARIZATION_STRENGTH:
                        print("E:" + str(e) + " MS:" + str(ms) + " LR:" + str(lr) + " RS:" + str(rs))
                        w = stochastic_gradient_descent(X_tr, ytr, e, ms, lr, rs)
                        new_fmse = fMSE(w, X_val, y_val)
                        print("\t fMSE: " + str(new_fmse))
                        if min_fmse > new_fmse:
                            optimal = {"E": e, "MB": ms, "LR": lr, "RS": rs}
                            print("new optimal" + str(optimal))
                            min_fmse = new_fmse

    w = stochastic_gradient_descent(X_tr, ytr, optimal['E'], optimal['MB'], optimal['LR'], optimal['RS'])
    showImage(w)
    # Report fMSE cost on the training and testing data (separately)
    print("fMSE training set:", fMSE(w, X_tr, ytr))
    print("fMSE validation set:", fMSE(w, X_val, y_val))
    print("fMSE testing set:", fMSE(w, X_te, yte))

train_age_regressor()